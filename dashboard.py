"""Control panel for the ppo_pong project.

Runs on the training box next to the repo:

    .venv/bin/python dashboard.py            # serves http://localhost:8787

Everything the CLI does, behind buttons: start/stop/resume training,
tournaments, TensorBoard, run inventory, play commands.
"""

import glob
import json
import os
import re
import shlex
import subprocess
import threading
import time

from flask import Flask, jsonify, request

PORT = 8787
REPO = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(REPO, ".venv", "bin", "python")
TRAIN_SESSION = "ppo"
TB_SESSION = "tb"
TRAIN_LOG = os.path.join(REPO, "train_vec.log")

app = Flask(__name__)

_ckpt_cache = {}
_tournament = {"running": False, "log": "", "results": None, "error": None}
_tournament_lock = threading.Lock()


def sh(cmd):
    return subprocess.run(
        cmd, cwd=REPO, capture_output=True, text=True, timeout=30
    )


def tmux_alive(session):
    return sh(["tmux", "has-session", "-t", session]).returncode == 0


def parse_train_log():
    """Extract the last progress line and recent promotions from the log."""
    if not os.path.exists(TRAIN_LOG):
        return {}
    with open(TRAIN_LOG, "rb") as f:
        f.seek(max(0, os.path.getsize(TRAIN_LOG) - 16384))
        text = f.read().decode(errors="replace")
    chunks = re.split(r"[\r\n]", text)
    stats = {}
    for chunk in reversed(chunks):
        m = re.search(
            r"update (\d+) episodes (\d+) avg score (-?[\d.]+) steps (\d+) "
            r"steps/s (\d+) promotions (\d+)",
            chunk,
        )
        if m:
            stats = {
                "update": int(m.group(1)),
                "episodes": int(m.group(2)),
                "avg_score": float(m.group(3)),
                "steps": int(m.group(4)),
                "steps_per_sec": int(m.group(5)),
                "promotions": int(m.group(6)),
            }
            break
    promos = [c for c in chunks if c.startswith("promotion ")]
    stats["recent_promotions"] = promos[-5:]
    return stats


def checkpoint_summary(path):
    try:
        mtime = os.path.getmtime(path)
        cached = _ckpt_cache.get(path)
        if cached and cached[0] == mtime:
            return cached[1]
        import torch

        state = torch.load(path, map_location="cpu", weights_only=False).get(
            "train_state", {}
        )
        summary = {
            "steps": state.get("n_steps", 0),
            "promotions": state.get("promotions", 0),
            "updates": state.get("update", 0),
        }
        _ckpt_cache[path] = (mtime, summary)
        return summary
    except Exception:
        return None


@app.get("/api/status")
def api_status():
    latest = None
    link = os.path.join(REPO, "runs", "latest")
    if os.path.islink(link):
        latest = os.path.basename(os.path.realpath(link))
    return jsonify(
        {
            "training": tmux_alive(TRAIN_SESSION),
            "tensorboard": tmux_alive(TB_SESSION),
            "latest_run": latest,
            "log": parse_train_log(),
            "tournament": {
                "running": _tournament["running"],
                "error": _tournament["error"],
            },
        }
    )


@app.get("/api/runs")
def api_runs():
    latest = os.path.realpath(os.path.join(REPO, "runs", "latest"))
    runs = []
    for d in sorted(glob.glob(os.path.join(REPO, "runs", "*"))):
        if not os.path.isdir(d) or os.path.islink(d):
            continue
        ckpt = os.path.join(d, "checkpoint.pt")
        has_ckpt = os.path.isfile(ckpt)
        runs.append(
            {
                "name": os.path.basename(d),
                "resumable": has_ckpt,
                "is_latest": os.path.realpath(d) == latest,
                "updated": time.strftime(
                    "%m-%d %H:%M", time.localtime(os.path.getmtime(d))
                ),
                "summary": checkpoint_summary(ckpt) if has_ckpt else None,
            }
        )
    runs.sort(key=lambda r: r["updated"], reverse=True)
    return jsonify(runs)


@app.get("/api/models")
def api_models():
    models = []
    for d in ["tmp/docker/models", "tmp/docker/goat_seed"]:
        if os.path.isdir(os.path.join(REPO, d)):
            models.append(d)
    for f in sorted(glob.glob(os.path.join(REPO, "tmp/docker/legacy_models/*"))):
        name = os.path.basename(f)
        if name.startswith("actor") and "copy" not in name:
            models.append(os.path.relpath(f, REPO))
    for d in sorted(glob.glob(os.path.join(REPO, "tmp/docker/legacy_models/*/"))):
        if glob.glob(os.path.join(d, "actor*")):
            models.append(os.path.relpath(d.rstrip("/"), REPO))
    for f in sorted(glob.glob(os.path.join(REPO, "runs/*/checkpoint.pt"))):
        models.append(os.path.relpath(f, REPO))
    return jsonify(models)


@app.post("/api/train/start")
def api_train_start():
    if tmux_alive(TRAIN_SESSION):
        return jsonify({"error": "training already running - stop it first"}), 409
    body = request.get_json(force=True)
    name = body.get("run_name", "").strip() or time.strftime("run_%Y%m%d_%H%M%S")
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", name):
        return jsonify({"error": "run name: letters, digits, _ and - only"}), 400
    if os.path.exists(os.path.join(REPO, "runs", name)):
        return jsonify({"error": f"runs/{name} already exists"}), 400
    num_envs = int(body.get("num_envs", 32))
    cmd = f"{PY} main_vec.py --num-envs {num_envs} --run-name {shlex.quote(name)}"
    init = body.get("init_weights", "").strip()
    if init:
        cmd += f" --init-weights {shlex.quote(init)}"
    r = sh(["tmux", "new-session", "-d", "-s", TRAIN_SESSION,
            f"cd {shlex.quote(REPO)} && {cmd} > train_vec.log 2>&1"])
    if r.returncode != 0:
        return jsonify({"error": r.stderr.strip()}), 500
    return jsonify({"ok": True, "run_name": name})


@app.post("/api/train/resume")
def api_train_resume():
    if tmux_alive(TRAIN_SESSION):
        return jsonify({"error": "training already running - stop it first"}), 409
    body = request.get_json(force=True)
    run = body.get("run", "").strip()
    arg = f" runs/{shlex.quote(run)}" if run else ""
    r = sh(["tmux", "new-session", "-d", "-s", TRAIN_SESSION,
            f"cd {shlex.quote(REPO)} && {PY} main_vec.py --resume{arg}"
            f" >> train_vec.log 2>&1"])
    if r.returncode != 0:
        return jsonify({"error": r.stderr.strip()}), 500
    return jsonify({"ok": True})


@app.post("/api/train/stop")
def api_train_stop():
    if not tmux_alive(TRAIN_SESSION):
        return jsonify({"error": "training is not running"}), 409
    sh(["tmux", "kill-session", "-t", TRAIN_SESSION])
    return jsonify({"ok": True})


@app.post("/api/tensorboard/start")
def api_tb_start():
    if tmux_alive(TB_SESSION):
        return jsonify({"ok": True, "note": "already running"})
    loop = (
        f"while true; do {REPO}/.venv/bin/tensorboard --logdir {REPO}/runs "
        f"--port 6006 --load_fast=false >> {REPO}/tb.log 2>&1; sleep 3; done"
    )
    r = sh(["tmux", "new-session", "-d", "-s", TB_SESSION, loop])
    if r.returncode != 0:
        return jsonify({"error": r.stderr.strip()}), 500
    return jsonify({"ok": True})


def run_tournament(models, points, legs):
    try:
        out = os.path.join("/tmp", "ui_tournament.json")
        proc = subprocess.run(
            [PY, "tournament.py", *models, "--points", str(points),
             "--legs", str(legs), "--output", out],
            cwd=REPO, capture_output=True, text=True, timeout=3600,
        )
        with _tournament_lock:
            _tournament["log"] = proc.stdout[-4000:]
            if proc.returncode == 0:
                with open(out) as f:
                    _tournament["results"] = json.load(f)
            else:
                _tournament["error"] = proc.stderr[-2000:] or "tournament failed"
    except Exception as exc:
        with _tournament_lock:
            _tournament["error"] = str(exc)
    finally:
        with _tournament_lock:
            _tournament["running"] = False


@app.post("/api/tournament")
def api_tournament_start():
    with _tournament_lock:
        if _tournament["running"]:
            return jsonify({"error": "a tournament is already running"}), 409
        body = request.get_json(force=True)
        models = body.get("models", [])
        if len(models) < 2:
            return jsonify({"error": "select at least two models"}), 400
        for m in models:
            full = os.path.realpath(os.path.join(REPO, m))
            if not full.startswith(os.path.realpath(REPO)) or not os.path.exists(full):
                return jsonify({"error": f"bad model path: {m}"}), 400
        _tournament.update(
            {"running": True, "log": "", "results": None, "error": None}
        )
    threading.Thread(
        target=run_tournament,
        args=(models, int(body.get("points", 5)), int(body.get("legs", 2))),
        daemon=True,
    ).start()
    return jsonify({"ok": True})


@app.get("/api/tournament")
def api_tournament_status():
    with _tournament_lock:
        return jsonify(_tournament)


@app.get("/")
def index():
    return PAGE


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PONG CONTROL</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#0a0d0b; --panel:#0e1210; --line:#1e2a22; --line2:#31473a;
  --txt:#c8d6cc; --dim:#5f7368; --green:#57f287; --amber:#f2c94c;
  --red:#ff5d5d; --glow:0 0 8px rgba(87,242,135,.45);
}
*{box-sizing:border-box;margin:0}
body{
  background:var(--bg); color:var(--txt);
  font:14px/1.5 "IBM Plex Mono",monospace;
  min-height:100vh; padding:28px 32px 64px;
  background-image:
    radial-gradient(ellipse 90% 60% at 50% -10%, rgba(87,242,135,.05), transparent),
    repeating-linear-gradient(0deg, transparent 0 2px, rgba(0,0,0,.18) 2px 3px);
}
h1{
  font-family:"Chakra Petch",sans-serif; font-size:26px; letter-spacing:.14em;
  color:var(--green); text-shadow:var(--glow);
}
h1 small{color:var(--dim);font-size:13px;letter-spacing:.28em;text-shadow:none}
header{display:flex;align-items:baseline;gap:22px;border-bottom:1px solid var(--line2);
  padding-bottom:14px;margin-bottom:22px;flex-wrap:wrap}
.leds{display:flex;gap:18px;margin-left:auto;font-size:11px;letter-spacing:.18em}
.led{display:flex;align-items:center;gap:7px;color:var(--dim)}
.led i{width:9px;height:9px;border-radius:50%;background:#333;display:inline-block}
.led.on{color:var(--txt)} .led.on i{background:var(--green);box-shadow:var(--glow);
  animation:pulse 2.2s infinite}
.led.off i{background:var(--red);box-shadow:0 0 8px rgba(255,93,93,.4)}
@keyframes pulse{50%{opacity:.55}}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px}
section{background:var(--panel);border:1px solid var(--line);padding:18px 20px;position:relative}
section::before{content:attr(data-title);position:absolute;top:-9px;left:12px;
  background:var(--bg);padding:0 8px;font-size:10.5px;letter-spacing:.3em;color:var(--dim)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:12px;margin-bottom:4px}
.tile{border:1px solid var(--line);padding:10px 12px}
.tile b{font-family:"Chakra Petch",sans-serif;font-size:24px;color:var(--green);display:block}
.tile span{font-size:10px;letter-spacing:.2em;color:var(--dim)}
button{
  background:transparent;border:1px solid var(--line2);color:var(--txt);
  font:600 12px "IBM Plex Mono",monospace;letter-spacing:.12em;
  padding:8px 16px;cursor:pointer;text-transform:uppercase;transition:.12s}
button:hover{background:var(--green);color:#06130a;border-color:var(--green);box-shadow:var(--glow)}
button.danger:hover{background:var(--red);border-color:var(--red);color:#140505;box-shadow:0 0 8px rgba(255,93,93,.5)}
button:disabled{opacity:.35;cursor:not-allowed}
button:disabled:hover{background:transparent;color:var(--txt);border-color:var(--line2);box-shadow:none}
input,select{background:#0a0f0c;border:1px solid var(--line);color:var(--txt);
  font:13px "IBM Plex Mono",monospace;padding:7px 9px;width:100%}
input:focus,select:focus{outline:none;border-color:var(--green)}
label{font-size:10px;letter-spacing:.2em;color:var(--dim);display:block;margin:10px 0 4px}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th{font-size:10px;letter-spacing:.2em;color:var(--dim);text-align:left;padding:6px 8px;border-bottom:1px solid var(--line2)}
td{padding:6px 8px;border-bottom:1px solid var(--line)}
tr:hover td{background:rgba(87,242,135,.04)}
.tag{font-size:10px;border:1px solid var(--line2);padding:1px 7px;letter-spacing:.15em;color:var(--amber)}
.msg{margin-top:10px;font-size:12px;min-height:18px}
.msg.err{color:var(--red)} .msg.ok{color:var(--green)}
.models{max-height:180px;overflow-y:auto;border:1px solid var(--line);padding:8px;margin-top:4px}
.models label{display:flex;gap:8px;align-items:center;margin:3px 0;font-size:12px;
  letter-spacing:0;color:var(--txt);cursor:pointer;text-transform:none}
.models input{width:auto}
pre{background:#070a08;border:1px solid var(--line);padding:10px;font-size:12px;
  overflow-x:auto;margin-top:8px;color:var(--amber)}
.row{display:flex;gap:10px;align-items:end;flex-wrap:wrap}
.row>div{flex:1;min-width:110px}
a{color:var(--green)}
.spin{display:inline-block;animation:rot 1s steps(8) infinite}
@keyframes rot{to{transform:rotate(360deg)}}
.promo{color:var(--amber);font-size:12px}
footer{margin-top:26px;color:var(--dim);font-size:11.5px;letter-spacing:.05em}
</style>
</head>
<body>
<header>
  <h1>PONG CONTROL <small>// SELF-PLAY TRAINING RIG</small></h1>
  <div class="leds">
    <span class="led" id="led-train"><i></i>TRAINING</span>
    <span class="led" id="led-tb"><i></i>TENSORBOARD</span>
  </div>
</header>

<section data-title="TELEMETRY" style="margin-bottom:18px">
  <div class="tiles" id="tiles"></div>
  <div id="run-line" style="font-size:12px;color:var(--dim)"></div>
  <div id="promos"></div>
</section>

<div class="grid">
<section data-title="TRAINING CONTROL">
  <div class="row" style="margin-bottom:14px">
    <button class="danger" id="btn-stop">&#9632; Stop</button>
    <button id="btn-resume">&#9654; Resume latest</button>
    <button id="btn-tb">Start TensorBoard</button>
  </div>
  <label>NEW RUN &mdash; NAME</label>
  <input id="run-name" placeholder="my_experiment (blank = timestamp)">
  <div class="row">
    <div><label>ENVS</label><input id="num-envs" type="number" value="32"></div>
    <div style="flex:2"><label>INIT WEIGHTS (OPTIONAL)</label>
      <select id="init-weights"><option value="">fresh random weights</option></select></div>
  </div>
  <div class="row" style="margin-top:12px">
    <button id="btn-start">&#9650; Launch new run</button>
  </div>
  <div class="msg" id="train-msg"></div>
</section>

<section data-title="TOURNAMENT">
  <label>COMBATANTS (PICK 2+)</label>
  <div class="models" id="model-list"></div>
  <div class="row" style="margin-top:10px">
    <div><label>POINTS / LEG</label><input id="t-points" type="number" value="5"></div>
    <div><label>LEGS</label><input id="t-legs" type="number" value="2"></div>
    <div><button id="btn-tournament">&#9876; Fight</button></div>
  </div>
  <div class="msg" id="t-msg"></div>
  <div id="t-results"></div>
</section>
</div>

<section data-title="RUNS" style="margin-top:18px">
  <table id="runs-table"><thead><tr>
    <th>RUN</th><th>STEPS</th><th>PROMOTIONS</th><th>UPDATED</th><th></th>
  </tr></thead><tbody></tbody></table>
</section>

<section data-title="PLAY VS HUMAN" style="margin-top:18px">
  <div class="row">
    <div style="flex:3"><label>OPPONENT MODEL</label>
      <select id="play-model"></select></div>
  </div>
  <pre id="play-cmd"></pre>
  <div style="font-size:11.5px;color:var(--dim)">run this in a terminal on the machine with the display &mdash; arrow keys, right paddle</div>
</section>

<footer>
  tunnel from your laptop: <b>ssh -L 8787:localhost:8787 -L 6006:localhost:6006 fedora</b>
  &nbsp;&middot;&nbsp; tensorboard: <a href="http://localhost:6006" target="_blank">localhost:6006</a>
</footer>

<script>
const $=id=>document.getElementById(id);
const fmt=n=>n>=1e6?(n/1e6).toFixed(2)+"M":n>=1e3?(n/1e3).toFixed(1)+"k":n;
async function api(path,opts){const r=await fetch(path,opts);const j=await r.json();
  if(!r.ok)throw new Error(j.error||r.statusText);return j}
function msg(id,text,ok){const el=$(id);el.textContent=text;el.className="msg "+(ok?"ok":"err");
  if(ok)setTimeout(()=>{el.textContent=""},6000)}

async function refreshStatus(){
  try{
    const s=await api("/api/status");
    $("led-train").className="led "+(s.training?"on":"off");
    $("led-tb").className="led "+(s.tensorboard?"on":"off");
    const l=s.log||{};
    const tiles=[["steps/s",l.steps_per_sec],["total steps",l.steps],
      ["promotions",l.promotions],["avg score",l.avg_score],
      ["episodes",l.episodes],["updates",l.update]];
    $("tiles").innerHTML=tiles.map(([k,v])=>
      `<div class="tile"><b>${v===undefined?"--":fmt(v)}</b><span>${k.toUpperCase()}</span></div>`).join("");
    $("run-line").textContent=s.latest_run?("latest run: "+s.latest_run+(s.training?"":" (stopped)")):"no runs yet";
    $("promos").innerHTML=(l.recent_promotions||[]).map(p=>`<div class="promo">&#9733; ${p}</div>`).join("");
    $("btn-stop").disabled=!s.training;
    $("btn-resume").disabled=s.training;
    $("btn-start").disabled=s.training;
  }catch(e){}
}

async function refreshRuns(){
  try{
    const runs=await api("/api/runs");
    $("runs-table").querySelector("tbody").innerHTML=runs.map(r=>{
      const s=r.summary||{};
      return `<tr><td>${r.name} ${r.is_latest?'<span class="tag">LATEST</span>':""}</td>
        <td>${s.steps!==undefined?fmt(s.steps):"--"}</td>
        <td>${s.promotions??"--"}</td><td>${r.updated}</td>
        <td>${r.resumable?`<button onclick="resumeRun('${r.name}')">resume</button>`:""}</td></tr>`;
    }).join("");
  }catch(e){}
}

async function refreshModels(){
  try{
    const models=await api("/api/models");
    $("model-list").innerHTML=models.map(m=>
      `<label><input type="checkbox" value="${m}">${m}</label>`).join("");
    $("init-weights").innerHTML='<option value="">fresh random weights</option>'+
      models.filter(m=>!m.endsWith(".pt")&&!m.startsWith("runs/")).map(m=>`<option>${m}</option>`).join("");
    $("play-model").innerHTML=models.map(m=>`<option>${m}</option>`).join("");
    updatePlayCmd();
  }catch(e){}
}
function updatePlayCmd(){
  $("play-cmd").textContent=`cd ~/src/ppo_pong\n.venv/bin/python play.py --model ${$("play-model").value||"..."}`;
}
$("play-model").onchange=updatePlayCmd;

$("btn-stop").onclick=async()=>{
  if(!confirm("Stop training? Progress is checkpointed every ~20s."))return;
  try{await api("/api/train/stop",{method:"POST"});msg("train-msg","training stopped",1);refreshStatus()}
  catch(e){msg("train-msg",e.message)}};
$("btn-resume").onclick=async()=>{
  try{await api("/api/train/resume",{method:"POST",headers:{"Content-Type":"application/json"},body:"{}"});
    msg("train-msg","resumed latest run",1);setTimeout(refreshStatus,1500)}
  catch(e){msg("train-msg",e.message)}};
$("btn-start").onclick=async()=>{
  try{const j=await api("/api/train/start",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({run_name:$("run-name").value,num_envs:+$("num-envs").value,
      init_weights:$("init-weights").value})});
    msg("train-msg","launched "+j.run_name,1);$("run-name").value="";
    setTimeout(()=>{refreshStatus();refreshRuns()},1500)}
  catch(e){msg("train-msg",e.message)}};
$("btn-tb").onclick=async()=>{
  try{await api("/api/tensorboard/start",{method:"POST"});msg("train-msg","tensorboard up",1);
    setTimeout(refreshStatus,1200)}catch(e){msg("train-msg",e.message)}};
window.resumeRun=async name=>{
  try{await api("/api/train/resume",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({run:name})});msg("train-msg","resumed "+name,1);setTimeout(refreshStatus,1500)}
  catch(e){msg("train-msg",e.message)}};

$("btn-tournament").onclick=async()=>{
  const models=[...document.querySelectorAll("#model-list input:checked")].map(c=>c.value);
  try{
    await api("/api/tournament",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({models,points:+$("t-points").value,legs:+$("t-legs").value})});
    msg("t-msg","tournament running...",1);pollTournament();
  }catch(e){msg("t-msg",e.message)}
};
async function pollTournament(){
  const t=await api("/api/tournament");
  if(t.running){$("t-msg").innerHTML='<span class="spin">&#10022;</span> fighting...';
    $("t-msg").className="msg ok";setTimeout(pollTournament,3000);return}
  if(t.error){msg("t-msg",t.error);return}
  if(!t.results){$("t-msg").textContent="";return}
  $("t-msg").textContent="";
  const st=t.results.standings;
  const rows=Object.entries(st).sort((a,b)=>b[1].wins-a[1].wins||
    (b[1].pf-b[1].pa)-(a[1].pf-a[1].pa));
  $("t-results").innerHTML=`<table><thead><tr><th>MODEL</th><th>W</th><th>D</th>
    <th>L</th><th>PF</th><th>PA</th></tr></thead><tbody>`+
    rows.map(([n,s])=>`<tr><td>${n}</td><td>${s.wins}</td><td>${s.draws}</td>
      <td>${s.losses}</td><td>${s.pf}</td><td>${s.pa}</td></tr>`).join("")+
    "</tbody></table>"+
    t.results.matches.map(m=>`<div class="promo">${m.match} &mdash; ${m.score.join(":")} &rarr; ${m.winner}</div>`).join("");
}

refreshStatus();refreshRuns();refreshModels();pollTournament();
setInterval(refreshStatus,3000);setInterval(refreshRuns,15000);
</script>
</body>
</html>"""


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=PORT)
