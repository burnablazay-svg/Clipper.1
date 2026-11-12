#!/usr/bin/env python3
"""Local web UI for clipper.py. Run: python app.py, then open http://127.0.0.1:5000"""
import json, os, re, secrets, shutil, subprocess, threading, time, uuid
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
                                            "style-src 'unsafe-inline'; media-src 'self'; img-src 'self' data:")
    return r


def run(job_id, url, count, focus, vertical, captions):
    job, out = JOBS[job_id], os.path.join(ROOT, job_id)
    try:
        os.makedirs(out, exist_ok=True)
        job["stage"] = "Downloading the video"
        src, heat, title = clipper.download(url, out)
        job["stage"] = ("Listening to the most-replayed sections" if heat
                        else "Transcribing the whole video (no replay heatmap, this is slower)")
        segs = clipper.transcribe(src, heat, count)
        job["stage"] = "Picking the best moments"
        clips = clipper.pick_moments(segs, heat, title, count, focus, src)
        vlen = clipper.video_length(heat, segs)
        for i, c in enumerate(clips, 1):
            job["stage"] = f"Cutting clip {i} of {len(clips)}"
            ps = max(0, c["start"] - clipper.PAD)
            pe = min(c["end"] + clipper.PAD, vlen) if vlen else c["end"] + clipper.PAD
            path = clipper.cut(src, {"start": ps, "end": pe}, i, out, segs, vertical, captions)
            dur = clipper.duration(path)
            job["clips"].append({"file": os.path.basename(path), "title": c["title"],
                                 "hook": c["hook"], "score": c["score"],
                                 "start": c["start"], "end": c["end"], "dur": dur,
                                 "mb": round(os.path.getsize(path) / 1e6, 1),
                                 "in": round(c["start"] - ps, 1),
                                 "out": round(min(c["end"] - ps, dur), 1)})
        json.dump(job["clips"], open(os.path.join(out, "clips.json"), "w"))
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
    if not re.fullmatch(r"[0-9a-f]{10}", job_id) or not name.endswith(".mp4"):
        return "Not found", 404
    return send_from_directory(os.path.join(ROOT, job_id), name)


@app.get("/disk")
def disk():
    os.makedirs(ROOT, exist_ok=True)
    return jsonify(free_gb=round(shutil.disk_usage(ROOT).free / 1e9, 1))


@app.get("/jobs")
def jobs():
    out = []
    for d in os.listdir(ROOT) if os.path.isdir(ROOT) else []:
        p = os.path.join(ROOT, d)
        if not re.fullmatch(r"[0-9a-f]{10}", d) or d in JOBS and JOBS[d]["status"] == "running":
            continue
        try:
            clips = json.load(open(os.path.join(p, "clips.json")))
        except Exception:
            clips = [{"file": f, "title": f, "hook": "", "score": 0, "start": 0, "end": 0}
                     for f in sorted(os.listdir(p)) if f.endswith(".mp4") and "_trim" not in f]
        size = {f: os.path.getsize(os.path.join(p, f)) for f in os.listdir(p)}
        for cl in clips:
            cl["mb"] = round(size.get(cl["file"], 0) / 1e6, 1)
        if clips:
            out.append({"id": d, "when": time.strftime("%d %b %H:%M", time.localtime(os.path.getmtime(p))),
                        "t": os.path.getmtime(p), "clips": clips, "mb": round(sum(size.values()) / 1e6, 1)})
    return jsonify(sorted(out, key=lambda j: -j["t"])[:8])


@app.post("/trim/<job_id>")
def trim(job_id):
    d = request.get_json(force=True)
    f = str(d.get("file", ""))
    if not re.fullmatch(r"[0-9a-f]{10}", job_id) or not re.fullmatch(r"clip\d{2}\.mp4", f):
        return jsonify(error="Bad request"), 400
    src = os.path.join(ROOT, job_id, f)
    try:
        a, b = float(d["start"]), float(d["end"])
    except (KeyError, TypeError, ValueError):
        return jsonify(error="Bad times"), 400
    if not os.path.exists(src) or a < 0 or b - a < 1:
        return jsonify(error="Bad range"), 400
    out = f.replace(".mp4", "_trim.mp4")
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(a), "-to", str(b), "-i", src,
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(max(16, clipper.CRF - 3)),
                        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", out],
                       check=True, cwd=os.path.join(ROOT, job_id))
    except subprocess.CalledProcessError:
        return jsonify(error="Could not trim this clip"), 500
    return jsonify(file=out)


@app.post("/delete/<job_id>")
def delete(job_id):
    if not re.fullmatch(r"[0-9a-f]{10}", job_id) or JOBS.get(job_id, {}).get("status") == "running":
        return jsonify(error="Cannot delete"), 400
    shutil.rmtree(os.path.join(ROOT, job_id), ignore_errors=True)
    JOBS.pop(job_id, None)
    return jsonify(ok=True)


@app.post("/open/<job_id>")
def open_folder(job_id):
    p = os.path.join(ROOT, job_id)
    if re.fullmatch(r"[0-9a-f]{10}", job_id) and os.path.isdir(p) and hasattr(os, "startfile"):
        os.startfile(p)
    return jsonify(ok=True)


PAGE = r"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Clipper</title>
<link rel=icon href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='14' fill='%23f2501a'/%3E%3Cpath d='M24 22v20l17-10z' fill='white'/%3E%3C/svg%3E">
<style>
:root{--peri:#8b8dff;--coral:#f2645e;--org:#f29a5f;--navy:#2a3f8f;--glass:rgba(12,14,46,.62);--line:rgba(255,255,255,.16)}
*{box-sizing:border-box}
body{margin:0;min-height:100vh;color:#fff;font:15px/1.5 "Segoe UI",system-ui,sans-serif;overflow-x:hidden;
background:linear-gradient(135deg,#1c2468 0%,#5b3fa8 42%,#e8607a 100%) fixed}
body:before{content:"";position:fixed;inset:0;background:radial-gradient(rgba(255,255,255,.5) 1px,transparent 1.5px) 0 0/30px 30px;opacity:.2;pointer-events:none}
.blob{position:fixed;width:62vw;height:62vw;left:22%;top:8%;background:radial-gradient(circle at 30% 30%,#5b86ff,#2a3aa8 55%,transparent 72%);
filter:blur(34px);opacity:.75;border-radius:42% 58% 63% 37%/45% 40% 60% 55%;animation:morph 16s ease-in-out infinite alternate;pointer-events:none}
@keyframes morph{to{border-radius:60% 40% 38% 62%/55% 62% 38% 45%;transform:translate(-4%,6%) rotate(24deg)}}
@media(prefers-reduced-motion:reduce){.blob{animation:none}}
.shell{position:relative;display:grid;grid-template-columns:92px 1fr;gap:28px;max-width:1180px;margin:0 auto;padding:28px 20px}
.rail{background:var(--glass);border:1px solid var(--line);border-radius:6px;padding:22px 0;display:flex;flex-direction:column;align-items:center;gap:22px;height:fit-content;position:sticky;top:28px;backdrop-filter:blur(8px)}
.st{display:flex;flex-direction:column;align-items:center;gap:8px;opacity:.45;transition:.3s}.st.on{opacity:1}.st.done{opacity:.8}
.st b{font:400 22px "Bebas Neue",Oswald,"Bahnschrift Condensed","Arial Narrow",Impact,sans-serif;color:var(--org);letter-spacing:1px}
.st span{writing-mode:vertical-rl;transform:rotate(180deg);font:12px "Arial Narrow",sans-serif;letter-spacing:3px;text-transform:uppercase}
.st:after{content:"";width:22px;height:2px;background:var(--org)}
.top{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.logo{letter-spacing:6px;font-size:13px;text-transform:uppercase;display:flex;align-items:center;gap:12px}
.chip{font-size:12px;padding:4px 12px;border:1px solid var(--line);border-radius:99px;background:var(--glass)}.chip.low{border-color:var(--coral);color:#ffb3ad}
h1{margin:6px 0 4px;font:400 clamp(64px,11vw,132px)/.9 "Bebas Neue",Oswald,"Bahnschrift Condensed",Impact,"Arial Narrow",sans-serif;letter-spacing:4px;text-transform:uppercase}
h1 span{color:var(--peri)}
.how{max-width:560px;margin:4px 0 20px}.how summary{cursor:pointer;display:inline-block;padding:8px 16px;border:1px solid var(--line);border-radius:99px;background:var(--glass);letter-spacing:1px;font-size:13px;text-transform:uppercase}.how ol{margin:10px 0 0;padding:14px 14px 14px 34px;background:var(--glass);border:1px solid var(--line);border-radius:6px;color:rgba(255,255,255,.88)}.sug{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}.sug button{font:inherit;font-size:12px;padding:4px 12px;border-radius:99px;border:1px solid var(--line);background:rgba(255,255,255,.08);color:#fff;cursor:pointer}.sug button:hover{border-color:var(--org)}
.panel{background:var(--glass);border:1px solid var(--line);border-radius:6px;padding:20px;backdrop-filter:blur(8px)}
label{display:block;font-size:12px;letter-spacing:2px;text-transform:uppercase;color:rgba(255,255,255,.7);margin:12px 0 5px}
input[type=text],input[type=number]{width:100%;padding:11px 12px;border:1px solid var(--line);border-radius:4px;background:rgba(255,255,255,.08);color:#fff;font:inherit}
input:focus{outline:2px solid var(--peri)}
.row{display:flex;gap:22px;flex-wrap:wrap;align-items:center;margin-top:16px}
.row label{display:inline;margin:0 0 0 6px;letter-spacing:1px;text-transform:none;font-size:14px;color:#fff}
.btn{padding:12px 26px;border:0;border-radius:3px;font:700 13px "Segoe UI",sans-serif;letter-spacing:2px;text-transform:uppercase;color:#fff;cursor:pointer;background:var(--coral)}
.btn:hover{filter:brightness(1.1)}.btn.alt{background:var(--navy)}.btn.sm{padding:6px 12px;font-size:11px}.btn[disabled]{opacity:.5;cursor:wait}
.err{color:#ffb3ad;min-height:20px;margin:8px 0 0;font-size:13px}
.num{display:flex;gap:18px;font:400 64px/1 "Bebas Neue",Oswald,"Bahnschrift Condensed",Impact,sans-serif;margin:26px 0 4px}
.num i{font-style:normal;color:transparent;-webkit-text-stroke:1px rgba(255,255,255,.6)}.num i.on{color:#fff;-webkit-text-stroke:0}
.bar{height:4px;background:rgba(255,255,255,.2);border-radius:2px;overflow:hidden}.bar div{height:100%;width:0;background:linear-gradient(90deg,var(--org),var(--coral));transition:width .6s}
#msg{margin:8px 0 18px;color:rgba(255,255,255,.85)}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px}
.card{background:var(--glass);border:1px solid var(--line);border-radius:6px;padding:10px}.card video{width:100%;border-radius:4px;background:#000}
.card h3{font:400 20px "Bebas Neue",Oswald,"Bahnschrift Condensed",Impact,sans-serif;letter-spacing:1px;margin:8px 0 2px}.card p{font-size:13px;color:rgba(255,255,255,.7);margin:0 0 8px}
.card a{color:var(--org);font-size:13px;letter-spacing:1px;text-transform:uppercase;text-decoration:none}
h2{font:400 28px "Bebas Neue",Oswald,"Bahnschrift Condensed",Impact,sans-serif;letter-spacing:3px;margin:30px 0 10px}
.hist{display:flex;gap:10px;align-items:center;justify-content:space-between;padding:9px 0;border-top:1px solid var(--line);flex-wrap:wrap}
@media(max-width:760px){.shell{grid-template-columns:1fr}.rail{flex-direction:row;justify-content:space-around;padding:12px;position:static}.st span{writing-mode:horizontal-tb;transform:none}.st:after{display:none}}
.trimbtn{margin-right:12px}.trim{margin-top:10px;padding-top:10px;border-top:1px solid var(--line)}.trim[hidden]{display:none}
.trk{position:relative;height:40px;background:rgba(255,255,255,.1);border-radius:6px}
.trk input{position:absolute;inset:0;width:100%;height:100%;margin:0;background:none;pointer-events:none;-webkit-appearance:none;appearance:none}
.trk input::-webkit-slider-runnable-track{background:none}.trk input::-moz-range-track{background:none}
.trk input::-webkit-slider-thumb{pointer-events:auto;-webkit-appearance:none;width:14px;height:40px;border-radius:4px;background:var(--org);border:2px solid #fff;box-sizing:border-box;cursor:ew-resize}
.trk input::-moz-range-thumb{pointer-events:auto;width:14px;height:40px;border-radius:4px;background:var(--org);border:2px solid #fff;box-sizing:border-box;cursor:ew-resize}
.sel{position:absolute;top:0;bottom:0;background:rgba(242,100,94,.35);border-top:3px solid var(--coral);border-bottom:3px solid var(--coral);pointer-events:none}
.ph{position:absolute;top:-5px;bottom:-5px;width:2px;background:#fff;pointer-events:none;z-index:3}
.tlr{display:flex;justify-content:space-between;margin-top:6px;font-size:12px;color:rgba(255,255,255,.85)}
.edbtn{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px}.edmsg{min-height:18px;margin:8px 0 0;font-size:12px;color:var(--org)}
</style>
<div class=blob></div>
<div class=shell>
<nav class=rail aria-label="Progress"><div class=st id=s1><b>01</b><span>Download</span></div><div class=st id=s2><b>02</b><span>Listen</span></div><div class=st id=s3><b>03</b><span>Pick</span></div><div class=st id=s4><b>04</b><span>Cut</span></div></nav>
<main>
<div class=top><div class=logo><svg width=38 height=38 viewBox="0 0 64 64" aria-hidden=true><defs><linearGradient id=lg x1=0 y1=0 x2=1 y2=1><stop offset=0 stop-color="#e8931c"/><stop offset=1 stop-color="#f23a0d"/></linearGradient></defs><rect width=64 height=64 rx=14 fill="url(#lg)"/><path d="M19 7v34a5 5 0 0 0 5 5h33M7 19h34a5 5 0 0 1 5 5v33" fill=none stroke=#fff stroke-width=5 stroke-linecap=round /><path d="M26 25v14l12-7z" fill=#fff stroke=#fff stroke-width=3 stroke-linejoin=round /></svg>Clipper</div><div class=chip id=disk>checking disk...</div></div>
<h1><span>Clip</span><br>Finder</h1>
<details class=how><summary>How does it work?</summary><ol>
<li>Paste a video link and press Find clips.</li>
<li>Clipper downloads it and looks for the most replayed moments. Pick a focus to look for something else.</li>
<li>Each moment is cut into a 45 to 60 second clip with about __PAD__ extra seconds on each side.</li>
<li>Open Trim length under any clip to set the exact start and end, then export it.</li></ol></details>
<div class=panel>
<label for=url>Video link</label><input id=url type=text placeholder="https://youtu.be/..." autocomplete=off>
<label for=focus>Focus (optional)</label><input id=focus type=text placeholder="Loudest moments" autocomplete=off><div class=sug id=sug></div>
<div class=row>
<span><label for=count style="margin:0 6px 0 0">Clips</label><input id=count type=number value=6 min=1 max=16 style="width:76px;display:inline-block"></span>
<span><input id=vertical type=checkbox checked><label for=vertical>9:16 vertical</label></span>
<span><input id=captions type=checkbox checked><label for=captions>Captions</label></span>
<button class=btn id=go>Find clips</button></div>
<p class=err id=err role=alert></p></div>
<div id=prog hidden><div class=num><i id=n1>01</i><i id=n2>02</i><i id=n3>03</i><i id=n4>04</i></div><div class=bar><div id=fill></div></div><div id=msg role=status></div></div>
<div id=grid></div>
<h2>Recent jobs <small id=tot style="font-size:15px;letter-spacing:1px;color:rgba(255,255,255,.7)"></small></h2><div id=hist class=panel>No earlier jobs yet.</div>
</main></div>
<script>
const $=id=>document.getElementById(id),H={'Content-Type':'application/json','X-Token':'__TOKEN__'};
let timer,shown=0,jid,t0;
function store(k,v){try{if(v===undefined)return localStorage.getItem(k);localStorage.setItem(k,v)}catch(e){return null}}
['focus','count'].forEach(k=>{const v=store('c_'+k);if(v!==null&&v!=='')$(k).value=v});
['vertical','captions'].forEach(k=>{const v=store('c_'+k);if(v!==null)$(k).checked=v==='1'});
async function disk(){try{const d=await(await fetch('/disk')).json();$('disk').textContent=d.free_gb+' GB free';
$('disk').className='chip'+(d.free_gb<3?' low':'');if(d.free_gb<3)$('disk').textContent='Low disk space: '+d.free_gb+' GB free'}catch(e){$('disk').textContent=''}}
function step(n,frac){for(let i=1;i<=4;i++){$('s'+i).className='st'+(i===n?' on':i<n?' done':'');$('n'+i).className=i<=n?'on':''}
$('fill').style.width=Math.round(((n-1)+(frac||0))/4*100)+'%'}
const fm=t=>Math.floor(t/60)+':'+(t%60).toFixed(1).padStart(4,'0');
function trimUI(c,id,v){
 const tp=document.createElement('div');tp.className='trim';tp.hidden=true;
 tp.innerHTML='<div class=trk><div class=sel></div><div class=ph></div><input type=range class=rs min=0 step=0.1 aria-label="Start"><input type=range class=re min=0 step=0.1 aria-label="End"></div><div class=tlr><span class=tin></span><span class=tlen></span><span class=tout></span></div><div class=edbtn><button class="btn alt sm pl">Play selection</button><button class="btn alt sm hs">Start here</button><button class="btn alt sm he">End here</button><button class="btn sm ex">Export</button></div><p class=edmsg role=status></p>';
 const q=s=>tp.querySelector(s),rs=q('.rs'),re=q('.re');let sw=false;
 rs.max=re.max=c.dur;rs.value=c.in;re.value=c.out;
 const px=x=>7+x/c.dur*(q('.trk').clientWidth-14);
 const upd=tp.upd=()=>{const a=+rs.value,b=+re.value;q('.sel').style.left=px(a)+'px';q('.sel').style.width=(px(b)-px(a))+'px';
  q('.tin').textContent='Start '+fm(a);q('.tout').textContent='End '+fm(b);q('.tlen').textContent='Length '+(b-a).toFixed(1)+'s'};
 const setA=a=>{rs.value=Math.max(0,Math.min(a,+re.value-1));upd()},setB=b=>{re.value=Math.min(c.dur,Math.max(b,+rs.value+1));upd()};
 rs.oninput=()=>{setA(+rs.value);v.currentTime=+rs.value};re.oninput=()=>{setB(+re.value);v.currentTime=+re.value};
 q('.hs').onclick=()=>setA(v.currentTime);q('.he').onclick=()=>setB(v.currentTime);
 q('.pl').onclick=()=>{v.currentTime=+rs.value;sw=true;v.play()};
 v.addEventListener('timeupdate',()=>{q('.ph').style.left=px(v.currentTime)+'px';if(sw&&v.currentTime>=+re.value){v.pause();sw=false}});
 v.addEventListener('pause',()=>{sw=false});
 q('.ex').onclick=async()=>{const x=q('.ex');x.disabled=true;q('.edmsg').textContent='Exporting...';
  const r=await fetch('/trim/'+id,{method:'POST',headers:H,body:JSON.stringify({file:c.file,start:+rs.value,end:+re.value})});
  const d=await r.json();x.disabled=false;if(!r.ok){q('.edmsg').textContent=d.error;return}
  const a=document.createElement('a');a.href='/files/'+id+'/'+d.file+'?t='+Date.now();a.download=d.file;document.body.append(a);a.click();a.remove();
  q('.edmsg').textContent='Saved as '+d.file+' and downloading now.'};
 window.addEventListener('resize',()=>{if(!tp.hidden)upd()});
 v.addEventListener('loadedmetadata',()=>{v.currentTime=c.in},{once:true});
 return tp}
function card(c,id){const u='/files/'+id+'/'+c.file,el=document.createElement('div');el.className='card';
const v=document.createElement('video');v.src=u;v.controls=true;v.preload='metadata';
const h=document.createElement('h3');h.textContent=c.title;
const p=document.createElement('p');p.textContent=c.hook+(c.end?' ('+Math.round(c.end-c.start)+'s, score '+c.score+')':'')+(c.mb?'  |  '+c.mb+' MB':'');
const a=document.createElement('a');a.href=u;a.download=c.file;a.textContent='Download full';el.append(v,h,p);
if(c.dur){const t=document.createElement('button');t.className='btn sm alt trimbtn';t.textContent='Trim length \u25BE';el.append(t);const tp=trimUI(c,id,v);el.append(tp);
t.onclick=()=>{tp.hidden=!tp.hidden;t.textContent='Trim length '+(tp.hidden?'\u25BE':'\u25B4');if(!tp.hidden)tp.upd()}}
el.append(a);$('grid').append(el)}
async function go(){
 const url=$('url').value.trim();if(!url){$('err').textContent='Paste a video link first';return}
 ['focus','count'].forEach(k=>store('c_'+k,$(k).value));['vertical','captions'].forEach(k=>store('c_'+k,$(k).checked?'1':'0'));
 clearInterval(timer);$('grid').innerHTML='';shown=0;$('err').textContent='';$('go').disabled=true;$('prog').hidden=false;step(1);$('msg').textContent='Starting...';
 const r=await fetch('/start',{method:'POST',headers:H,body:JSON.stringify({url,focus:$('focus').value,count:+$('count').value,vertical:$('vertical').checked,captions:$('captions').checked})});
 const d=await r.json();if(!r.ok){$('err').textContent=d.error;$('go').disabled=false;$('prog').hidden=true;return}
 jid=d.id;t0=Date.now();timer=setInterval(poll,2000)}
async function poll(){
 const j=await(await fetch('/status/'+jid)).json(),m=Math.floor((Date.now()-t0)/60000),s=Math.floor((Date.now()-t0)/1000)%60;
 if(j.status==='error'){clearInterval(timer);$('err').textContent=j.error;$('go').disabled=false;$('prog').hidden=true;return}
 const g=j.stage||'',c=/Cutting clip (\d+) of (\d+)/.exec(g);
 step(c?4:g.startsWith('Picking')?3:/^(Listening|Transcribing)/.test(g)?2:1,c?(c[1]-1)/c[2]:0);
 $('msg').textContent=g+'  ('+m+':'+String(s).padStart(2,'0')+')';
 for(;shown<j.clips.length;shown++)card(j.clips[shown],jid);
 if(j.status==='done'){clearInterval(timer);step(5);$('fill').style.width='100%';$('msg').textContent=j.clips.length+' clips ready in '+m+':'+String(s).padStart(2,'0');$('go').disabled=false;disk();hist()}}
async function hist(){
 const L=await(await fetch('/jobs')).json();if(!L.length){$('hist').textContent='No earlier jobs yet.';$('tot').textContent='';return}
 $('tot').textContent='Total '+L.reduce((s,j)=>s+j.mb,0).toFixed(1)+' MB';
 $('hist').innerHTML='';L.forEach(j=>{const r=document.createElement('div');r.className='hist';
 const t=document.createElement('span');t.textContent=j.when+'  -  '+j.clips.length+' clips  -  '+j.mb+' MB';r.append(t);
 const b=document.createElement('span');
 [['View',()=>{$('grid').innerHTML='';j.clips.forEach(c=>card(c,j.id));$('grid').scrollIntoView({behavior:'smooth'})},'alt'],
  ['Open folder',()=>fetch('/open/'+j.id,{method:'POST',headers:H}),'alt'],
  ['Delete',async()=>{if(confirm('Delete these clips from your disk?')){await fetch('/delete/'+j.id,{method:'POST',headers:H});hist();disk()}},'']].forEach(([n,f,k])=>{
  const x=document.createElement('button');x.className='btn sm '+k;x.textContent=n;x.style.marginLeft='6px';x.onclick=f;b.append(x)});
 r.append(b);$('hist').append(r)})}
const SUG=['Loudest moments','Biggest donations','Most replayed','Trending'];let si=0;
SUG.forEach(n=>{const b=document.createElement('button');b.textContent=n;b.onclick=()=>{$('focus').value=n};$('sug').append(b)});
setInterval(()=>{si=(si+1)%SUG.length;$('focus').placeholder=SUG[si]},2500);
$('go').onclick=go;$('url').addEventListener('keydown',e=>{if(e.key==='Enter')go()});
disk();hist();
</script>"""


@app.get("/")
def index():
    return PAGE.replace("__TOKEN__", TOKEN).replace("__PAD__", str(clipper.PAD))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, threaded=True)