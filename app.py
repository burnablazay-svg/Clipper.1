#!/usr/bin/env python3
"""Local web UI for clipper.py. Run: python app.py, then open http://127.0.0.1:5000"""
import json, os, re, secrets, shutil, threading, time, uuid
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
        job["stage"] = ("Listening to the most-replayed sections" if heat
                        else "Transcribing the whole video (no replay heatmap, this is slower)")
        segs = clipper.transcribe(src, heat, count)
        job["stage"] = "Picking the best moments"
        clips = clipper.pick_moments(segs, heat, title, count, focus)
        for i, c in enumerate(clips, 1):
            job["stage"] = f"Cutting clip {i} of {len(clips)}"
            path = clipper.cut(src, c, i, out, segs, vertical, captions)
            job["clips"].append({"file": os.path.basename(path), "title": c["title"],
                                 "hook": c["hook"], "score": c["score"],
                                 "start": c["start"], "end": c["end"]})
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
                     for f in sorted(os.listdir(p)) if f.endswith(".mp4")]
        if clips:
            out.append({"id": d, "when": time.strftime("%d %b %H:%M", time.localtime(os.path.getmtime(p))),
                        "t": os.path.getmtime(p), "clips": clips})
    return jsonify(sorted(out, key=lambda j: -j["t"])[:8])


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
.logo{letter-spacing:6px;font-size:13px;text-transform:uppercase}
.chip{font-size:12px;padding:4px 12px;border:1px solid var(--line);border-radius:99px;background:var(--glass)}.chip.low{border-color:var(--coral);color:#ffb3ad}
h1{margin:6px 0 4px;font:400 clamp(64px,11vw,132px)/.9 "Bebas Neue",Oswald,"Bahnschrift Condensed",Impact,"Arial Narrow",sans-serif;letter-spacing:4px;text-transform:uppercase}
h1 span{color:var(--peri)}
.lead{max-width:520px;color:rgba(255,255,255,.82);margin:0 0 20px}
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
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:16px}
.card{background:var(--glass);border:1px solid var(--line);border-radius:6px;padding:10px}.card video{width:100%;border-radius:4px;background:#000}
.card h3{font:400 20px "Bebas Neue",Oswald,"Bahnschrift Condensed",Impact,sans-serif;letter-spacing:1px;margin:8px 0 2px}.card p{font-size:13px;color:rgba(255,255,255,.7);margin:0 0 8px}
.card a{color:var(--org);font-size:13px;letter-spacing:1px;text-transform:uppercase;text-decoration:none}
h2{font:400 28px "Bebas Neue",Oswald,"Bahnschrift Condensed",Impact,sans-serif;letter-spacing:3px;margin:30px 0 10px}
.hist{display:flex;gap:10px;align-items:center;justify-content:space-between;padding:9px 0;border-top:1px solid var(--line);flex-wrap:wrap}
@media(max-width:760px){.shell{grid-template-columns:1fr}.rail{flex-direction:row;justify-content:space-around;padding:12px;position:static}.st span{writing-mode:horizontal-tb;transform:none}.st:after{display:none}}
</style>
<div class=blob></div>
<div class=shell>
<nav class=rail aria-label="Progress"><div class=st id=s1><b>01</b><span>Download</span></div><div class=st id=s2><b>02</b><span>Listen</span></div><div class=st id=s3><b>03</b><span>Pick</span></div><div class=st id=s4><b>04</b><span>Cut</span></div></nav>
<main>
<div class=top><div class=logo>Clipper</div><div class=chip id=disk>checking disk...</div></div>
<h1><span>Clip</span><br>Finder</h1>
<p class=lead>Paste a video link. Clipper finds the most replayed moments and cuts them into 45 to 60 second clips.</p>
<div class=panel>
<label for=url>Video link</label><input id=url type=text placeholder="https://youtu.be/..." autocomplete=off>
<label for=focus>Focus (optional, needs an API key)</label><input id=focus type=text placeholder="The funniest reactions">
<div class=row>
<span><label for=count style="margin:0 6px 0 0">Clips</label><input id=count type=number value=6 min=1 max=16 style="width:76px;display:inline-block"></span>
<span><input id=vertical type=checkbox checked><label for=vertical>9:16 vertical</label></span>
<span><input id=captions type=checkbox checked><label for=captions>Captions</label></span>
<button class=btn id=go>Find clips</button></div>
<p class=err id=err role=alert></p></div>
<div id=prog hidden><div class=num><i id=n1>01</i><i id=n2>02</i><i id=n3>03</i><i id=n4>04</i></div><div class=bar><div id=fill></div></div><div id=msg role=status></div></div>
<div id=grid></div>
<h2>Recent jobs</h2><div id=hist class=panel>No earlier jobs yet.</div>
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
function card(c,id){const u='/files/'+id+'/'+c.file,el=document.createElement('div');el.className='card';
const v=document.createElement('video');v.src=u;v.controls=true;v.preload='metadata';
const h=document.createElement('h3');h.textContent=c.title;
const p=document.createElement('p');p.textContent=c.hook+(c.end?' ('+Math.round(c.end-c.start)+'s, score '+c.score+')':'');
const a=document.createElement('a');a.href=u;a.download=c.file;a.textContent='Download';el.append(v,h,p,a);$('grid').append(el)}
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
 const L=await(await fetch('/jobs')).json();if(!L.length){$('hist').textContent='No earlier jobs yet.';return}
 $('hist').innerHTML='';L.forEach(j=>{const r=document.createElement('div');r.className='hist';
 const t=document.createElement('span');t.textContent=j.when+'  -  '+j.clips.length+' clips';r.append(t);
 const b=document.createElement('span');
 [['View',()=>{$('grid').innerHTML='';j.clips.forEach(c=>card(c,j.id));$('grid').scrollIntoView({behavior:'smooth'})},'alt'],
  ['Open folder',()=>fetch('/open/'+j.id,{method:'POST',headers:H}),'alt'],
  ['Delete',async()=>{if(confirm('Delete these clips from your disk?')){await fetch('/delete/'+j.id,{method:'POST',headers:H});hist();disk()}},'']].forEach(([n,f,k])=>{
  const x=document.createElement('button');x.className='btn sm '+k;x.textContent=n;x.style.marginLeft='6px';x.onclick=f;b.append(x)});
 r.append(b);$('hist').append(r)})}
$('go').onclick=go;$('url').addEventListener('keydown',e=>{if(e.key==='Enter')go()});
disk();hist();
</script>"""


@app.get("/")
def index():
    return PAGE.replace("__TOKEN__", TOKEN)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, threaded=True)