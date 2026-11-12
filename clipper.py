#!/usr/bin/env python3
"""Standalone clipping agent: link in, viral clips out. No Everygen needed.

Pipeline: yt-dlp (download + YouTube "most replayed" heatmap) -> faster-whisper
(transcript) -> Claude (picks moments) -> ffmpeg (cut, optional 9:16, captions).
"""
import argparse, json, os, re, subprocess, sys
import numpy as np
import yt_dlp
from faster_whisper import WhisperModel
import anthropic

MODEL = os.environ.get("CLIPPER_MODEL", "claude-sonnet-5-5")

# Storage vs quality. Lower numbers or higher CRF = smaller files.
MAX_HEIGHT = 720          # source download height (1080 for sharper, larger files)
CRF = 24                  # x264 quality: 18 sharp/large ... 28 small/soft
VERT_W, VERT_H = 720, 1280  # 9:16 clip size (1080x1920 for full HD)
# Speed. "tiny" finds moments fast; "base" writes the caption words more accurately.
PICK_MODEL = os.environ.get("CLIPPER_PICK_MODEL", "tiny")
CAP_MODEL = os.environ.get("CLIPPER_CAPTION_MODEL", "base")
_MODELS = {}

RUBRIC = """You pick short-form clips from a video transcript.
Good clip: 15-40s (up to 60s if compiling). Setup of 2-8s so a cold viewer
understands, a twist or escalation around 6-10s, the peak reaction, then a hard
cut at or just after the peak. Favor sudden shifts, big reactions, surprises,
physical comedy, strong claims. Avoid inside jokes and slow explanations.
Use the heatmap (0-1 replay intensity by time) as strong evidence of what
viewers rewatched. Clips must not overlap. Return ONLY JSON:
[{"start": sec, "end": sec, "title": str, "hook": str, "score": 1-10}]"""


def download(url, workdir):
    h = f"[height<={MAX_HEIGHT}]"
    opts = {"format": f"bv*[ext=mp4][vcodec^=avc1]{h}+ba[ext=m4a]/b[ext=mp4]{h}/bv*{h}+ba/b",
            "merge_output_format": "mp4",
            "outtmpl": os.path.join(workdir, "source.%(ext)s"), "quiet": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    return os.path.join(workdir, "source.mp4"), info.get("heatmap") or [], info.get("title", "")


def get_model(name):
    if name not in _MODELS:
        _MODELS[name] = WhisperModel(name, compute_type="int8")
    return _MODELS[name]


def load_audio(path, start=0, dur=None):
    """Decode audio (optionally just one section) to 16 kHz mono float32 with ffmpeg."""
    cmd = ["ffmpeg", "-v", "error"]
    if start:
        cmd += ["-ss", str(start)]
    if dur:
        cmd += ["-t", str(dur)]
    cmd += ["-i", path, "-f", "f32le", "-ac", "1", "-ar", "16000", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, dtype=np.float32)


def heat_windows(heat, count, pad=35):
    """Time ranges around the most-replayed peaks, so we only transcribe those."""
    peaks = sorted(heat, key=lambda h: -h["value"])[:max(count * 3, 10)]
    spans = sorted((max(0, (p["start_time"] + p["end_time"]) / 2 - pad),
                    (p["start_time"] + p["end_time"]) / 2 + pad) for p in peaks)
    merged = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def transcribe(path, heat=None, count=6):
    """Transcribe quickly. With a replay heatmap, only the busiest sections are processed."""
    model = get_model(PICK_MODEL)
    wins = heat_windows(heat, count) if heat else [[0, None]]
    segs = []
    for a, b in wins:
        audio = load_audio(path, a, None if b is None else b - a)
        it, _ = model.transcribe(audio, vad_filter=True, beam_size=1,
                                 condition_on_previous_text=False)
        segs += [{"start": round(s.start + a, 1), "end": round(s.end + a, 1),
                  "text": s.text.strip()} for s in it]
    return segs


HYPE = ("wow", "oh my", "no way", "what", "haha", "crazy", "insane", "omg", "oh no",
        "unbelievable", "scream", "can't believe", "are you serious")


def pick_local(segments, heatmap, count, win=25):
    """No API key needed: replay heatmap plus simple transcript energy signals."""
    if not segments:
        return []
    total = max(s["end"] for s in segments)

    def heat_at(a, b):
        v = [h["value"] for h in heatmap if h["end_time"] > a and h["start_time"] < b]
        return max(v) if v else 0

    def text_score(a, b):
        txt = " ".join(s["text"] for s in segments if s["end"] > a and s["start"] < b).lower()
        return (len(txt.split()) / (b - a) + txt.count("!") * 0.5 + txt.count("?") * 0.2
                + sum(txt.count(w) for w in HYPE) * 0.7)

    win = min(win, max(total, 5))
    cands, t = [], 0.0
    while t + win <= total + 0.01:
        cands.append((heat_at(t, t + win) * 10 + text_score(t, t + win), t))
        t += 5
    picked = []
    for sc, t in sorted(cands, reverse=True):
        if all(abs(t - p["start"]) >= win for p in picked):
            seg = [s for s in segments if s["end"] > t and s["start"] < t + win]
            first = seg[0]["text"] if seg else ""
            picked.append({"start": t, "end": t + win, "title": first[:50] or "Moment",
                           "hook": first[:80], "score": min(10, max(1, round(sc)))})
        if len(picked) >= count:
            break
    return picked


def pick_moments(segments, heatmap, title, count, focus):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("No ANTHROPIC_API_KEY set: using the local picker.")
        return pick_local(segments, heatmap, count)
    peaks = sorted(heatmap, key=lambda h: -h["value"])[:20]
    heat = [[round(p["start_time"]), round(p["end_time"]), round(p["value"], 2)] for p in peaks]
    lines = "\n".join(f"[{s['start']}-{s['end']}] {s['text']}" for s in segments)[:300000]
    ask = f"Video: {title}\nFind up to {count} clips."
    if focus:
        ask += f" Focus: {focus}."
    ask += f"\nHeatmap peaks [start,end,intensity]: {json.dumps(heat)}\nTranscript:\n{lines}"
    msg = anthropic.Anthropic().messages.create(
        model=MODEL, max_tokens=4000, system=RUBRIC,
        messages=[{"role": "user", "content": ask}])
    text = "".join(b.text for b in msg.content if b.type == "text")
    clips = json.loads(re.search(r"\[.*\]", text, re.S).group(0))
    clips = [c for c in clips if 5 <= c["end"] - c["start"] <= 90]
    return sorted(clips, key=lambda c: -c["score"])[:count]


def ts(t):
    h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
    return f"{h:02}:{m:02}:{s:06.3f}".replace(".", ",")


def caption_chunks(src, start, end, max_words=4):
    """Re-transcribe just this clip with word timings, grouped into short caption lines."""
    try:
        audio = load_audio(src, start, end - start)
        it, _ = get_model(CAP_MODEL).transcribe(audio, word_timestamps=True, beam_size=1)
        words = [w for seg in it for w in (seg.words or [])]
    except Exception:
        return []
    groups, cur = [], []
    for w in words:
        t = w.word.strip()
        if not t:
            continue
        if cur and (len(cur) >= max_words or w.start - cur[-1][2] > 0.7
                    or cur[-1][0].endswith((".", "?", "!"))):
            groups.append(cur)
            cur = []
        cur.append((t, w.start, w.end))
    if cur:
        groups.append(cur)
    return [{"start": g[0][1], "end": max(g[-1][2], g[0][1] + 0.3),
             "text": " ".join(x[0] for x in g)} for g in groups]


def write_srt(chunks, path):
    with open(path, "w") as f:
        for n, c in enumerate(chunks, 1):
            f.write(f"{n}\n{ts(c['start'])} --> {ts(c['end'])}\n{c['text'].upper()}\n\n")
    return len(chunks)


def cut(src, clip, idx, outdir, segments, vertical, captions):
    name = f"clip{idx:02}"
    out, srt = f"{name}.mp4", f"{name}.srt"
    vf = []
    if vertical:
        vf.append(f"split[a][b];[a]scale={VERT_W}:{VERT_H}:force_original_aspect_ratio=increase,"
                  f"crop={VERT_W}:{VERT_H},scale={VERT_W // 20}:{VERT_H // 20},"
                  f"scale={VERT_W}:{VERT_H}:flags=bilinear[bg];[b]scale={VERT_W}:-2[fg];"
                  "[bg][fg]overlay=(W-w)/2:(H-h)/2")
    if captions and write_srt(caption_chunks(src, clip["start"], clip["end"]), os.path.join(outdir, srt)):
        # libass sizes are relative to a 288-pixel-high canvas, scaled to the video
        style = ("Fontsize=12,Bold=1,Outline=2,Shadow=0,Alignment=2,MarginL=20,MarginR=20,MarginV=60"
                 if vertical else "Fontsize=10,Bold=1,Outline=2,Shadow=0,Alignment=2,MarginV=10")
        vf.append(f"subtitles={srt}:force_style='{style}'")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(clip["start"]), "-to", str(clip["end"]),
           "-i", os.path.abspath(src)]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    subprocess.run(cmd + ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(CRF),
                  "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", out], check=True, cwd=outdir)
    return os.path.join(outdir, out)


def cleanup(outdir):
    """Delete the big source download and temporary caption files."""
    for f in os.listdir(outdir):
        if f.startswith("source") or f.endswith(".srt"):
            try:
                os.remove(os.path.join(outdir, f))
            except OSError:
                pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("url")
    p.add_argument("--count", type=int, default=6)
    p.add_argument("--focus", default="", help="e.g. 'the funniest reactions'")
    p.add_argument("--vertical", action="store_true", help="9:16 with blurred background")
    p.add_argument("--captions", action="store_true", help="burn in captions")
    p.add_argument("--out", default="clips")
    p.add_argument("--keep-source", action="store_true", help="keep the downloaded video")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print("Downloading...")
    src, heat, title = download(a.url, a.out)
    print(f"Transcribing... (heatmap: {'yes' if heat else 'not available'})")
    segs = transcribe(src, heat, a.count)
    print("Picking moments...")
    clips = pick_moments(segs, heat, title, a.count, a.focus)
    for i, c in enumerate(clips, 1):
        path = cut(src, c, i, a.out, segs, a.vertical, a.captions)
        print(f"{path}  [{c['start']:.0f}-{c['end']:.0f}s] score {c['score']}  {c['title']} | {c['hook']}")
    json.dump(clips, open(os.path.join(a.out, "clips.json"), "w"), indent=2)
    if not a.keep_source:
        cleanup(a.out)


if __name__ == "__main__":
    sys.exit(main())