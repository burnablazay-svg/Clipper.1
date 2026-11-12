#!/usr/bin/env python3
"""Local web UI for clipper.py. Run: python app.py, then open http://127.0.0.1:5000"""
import os, re, secrets, threading, uuid
from urllib.parse import urlparse
from flask import Flask, jsonify, request, send_from_directory
import clipper

app = Flask(__name__)
ROOT = os.path.abspath("jobs")
JOBS = {}
TOKEN = secrets.token_urlsafe(24)
HOSTS = {"127.0.0.1:5000", "localhost:5000"}
SITES = ("youtube.com", "youtu.be", "twitch.tv", "kick.com", "vimeo.com", "tiktok.com",
         "instagram.com", "x.com", "twitter.com", "facebook.com")
LOCK = threading.Lock()


def site_ok(url):
    u = urlparse(url)
    h = (u.hostname or "").lower()
    return u.scheme in ("http", "https") and any(h == s or h.endswith("." + s) for s in SITES)


@app.before_request
def guard():
    if request.host not in HOSTS:
        return jsonify(error="Bad host"), 403
    if request.method == "POST":
        o = request.headers.get("Origin")
        if (o and urlparse(o).netloc not in HOSTS) or request.headers.get("X-Token") != TOKEN:
            return jsonify(error="Request blocked"), 403


@app.after_request
def headers(r):
    r.headers["X-Content-Type-Options"] = "nosniff"
    r.headers["X-Frame-Options"] = "DENY"
    r.headers["Referrer-Policy"] = "no-referrer"
    r.headers["Content-Security-Policy"] = ("default-src 'self'; script-src 'unsafe-inline'; "
                                            "style-src 'unsafe-inline'; media-src 'self'")
    return r


def run(job_id, url, count, focus, vertical, captions):
    job, out = JOBS[job_id], os.path.join(ROOT, job_id)
    try:
        os.makedirs(out, exist_ok=True)
        job["stage"] = "Downloading the video"
        src, heat, title = clipper.download(url, out)
        job["stage"] = "Transcribing" + ("" if heat else " (no replay heatmap for this video)")
        segs = clipper.transcribe(src)
        job["stage"] = "Picking the best moments"
        clips = clipper.pick_moments(segs, heat, title, count, focus)
        for i, c in enumerate(clips, 1):
            job["stage"] = f"Cutting clip {i} of {len(clips)}"
            path = clipper.cut(src, c, i, out, segs, vertical, captions)
            job["clips"].append({"file": os.path.basename(path), "title": c["title"],
                                 "hook": c["hook"], "score": c["score"],
                                 "start": c["start"], "end": c["end"]})
        job["status"], job["stage"] = "done", "Done"
    except Exception as e:
        job["status"], job["error"] = "error", str(e)
    finally:
        clipper.cleanup(out)


@app.post("/start")
def start():
    d = request.get_json(force=True)
    url = (d.get("url") or "").strip()
    if not site_ok(url):
        return jsonify(error="Use a link from YouTube, Twitch, Kick, Vimeo, TikTok, Instagram, X, or Facebook"), 400
    if any(j["status"] == "running" for j in JOBS.values()):
        return jsonify(error="A job is already running. Wait for it to finish."), 429
    job_id = uuid.uuid4().hex[:10]
    JOBS[job_id] = {"status": "running", "stage": "Starting", "clips": [], "error": ""}
    count = max(1, min(int(d.get("count") or 6), 16))
    threading.Thread(target=run, daemon=True, args=(
        job_id, url, count, (d.get("focus") or "").strip(),
        bool(d.get("vertical")), bool(d.get("captions")))).start()
    return jsonify(id=job_id)


@app.get("/status/<job_id>")
def status(job_id):
    if not re.fullmatch(r"[0-9a-f]{10}", job_id):
        return jsonify(status="error", error="Unknown job"), 404
    return jsonify(JOBS.get(job_id, {"status": "error", "error": "Unknown job"}))


@app.get("/files/<job_id>/<path:name>")
def files(job_id, name):
    if job_id not in JOBS or not name.endswith(".mp4"):
        return "Not found", 404
    return send_from_directory(os.path.join(ROOT, job_id), name)


PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Clipper</title>
<style>
:root{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#ddd;--acc:#2563eb}
@media(prefers-color-scheme:dark){:root{--bg:#161616;--fg:#eee;--mut:#999;--line:#333;--acc:#6b9bff}}
body{font:16px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--fg);max-width:880px;margin:2rem auto;padding:0 1rem}
h1{font-size:22px;font-weight:500}label{display:block;font-size:13px;color:var(--mut);margin:12px 0 4px}
input[type=text],input[type=number]{width:100%;box-sizing:border-box;padding:8px 10px;border:1px solid var(--line);border-radius:8px;background:transparent;color:var(--fg);font:inherit}
.row{display:flex;gap:16px;flex-wrap:wrap;align-items:center;margin-top:12px}.row label{display:inline;margin:0;color:var(--fg)}
button{padding:8px 16px;border:1px solid var(--acc);background:var(--acc);color:#fff;border-radius:8px;font:inherit;cursor:pointer}
#msg{margin:16px 0;color:var(--mut)}#msg.err{color:#d33}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:16px}
.card{border:1px solid var(--line);border-radius:12px;padding:12px}.card video{width:100%;border-radius:8px;background:#000}
.card h3{font-size:15px;font-weight:500;margin:8px 0 2px}.card p{font-size:13px;color:var(--mut);margin:0 0 6px}
</style>
<h1>Clipper</h1>
<label for=url>Video link</label><input id=url type=text placeholder="https://youtu.be/...">
<label for=focus>Focus (optional)</label><input id=focus type=text placeholder="The funniest reactions">
<div class=row>
<span><label for=count style="display:inline">Clips</label> <input id=count type=number value=6 min=1 max=16 style="width:70px"></span>
<span><input id=vertical type=checkbox checked> <label for=vertical>9:16 vertical</label></span>
<span><input id=captions type=checkbox checked> <label for=captions>Captions</label></span>
<button id=go>Find clips</button></div>
<div id=msg role=status></div><div id=grid></div>
<script>
const $=id=>document.getElementById(id);let timer,shown=0,jid;
function say(t,e){$('msg').textContent=t;$('msg').className=e?'err':''}
$('go').onclick=async()=>{
 clearInterval(timer);$('grid').innerHTML='';shown=0;say('Starting...');
 const r=await fetch('/start',{method:'POST',headers:{'Content-Type':'application/json','X-Token':'__TOKEN__'},body:JSON.stringify({
  url:$('url').value,focus:$('focus').value,count:+$('count').value,vertical:$('vertical').checked,captions:$('captions').checked})});
 const d=await r.json();if(!r.ok){say(d.error,1);return}
 jid=d.id;timer=setInterval(poll,2000)};
async function poll(){
 const j=await(await fetch('/status/'+jid)).json();
 if(j.status==='error'){clearInterval(timer);say(j.error,1);return}
 say(j.stage);
 for(;shown<j.clips.length;shown++){const c=j.clips[shown],u='/files/'+jid+'/'+c.file;
  const el=document.createElement('div');el.className='card';
  const v=document.createElement('video');v.src=u;v.controls=true;v.preload='metadata';
  const h=document.createElement('h3');h.textContent=c.title;
  const p=document.createElement('p');p.textContent=c.hook+' ('+Math.round(c.start)+'s to '+Math.round(c.end)+'s, score '+c.score+')';
  const a=document.createElement('a');a.href=u;a.download=c.file;a.textContent='Download';
  el.append(v,h,p,a);$('grid').append(el)}
 if(j.status==='done'){clearInterval(timer);say(j.clips.length+' clips ready')}}
</script>"""


@app.get("/")
def index():
    return PAGE.replace("__TOKEN__", TOKEN)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, threaded=True)