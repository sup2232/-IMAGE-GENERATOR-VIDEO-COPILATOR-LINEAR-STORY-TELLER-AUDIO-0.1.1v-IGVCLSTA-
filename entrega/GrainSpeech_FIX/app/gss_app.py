"""
GrainSpeech Studio — aplicativo desktop (Windows/Linux/macOS).

Sobe um servidor HTTP local com a interface e abre o navegador. Tudo roda em
máquina própria; nenhum dado sai do computador (os modelos são baixados uma
única vez para um cache local).

    python gss_app.py --project C:\\GrainSpeechStudio\\meuprojeto \\
                      --repo   C:\\GrainSpeechStudio\\GrainSpeech

Empacotado:  GrainSpeechStudio.exe  (mesma linha de comando, sem `python`)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

IS_FROZEN = getattr(sys, "frozen", False)
PY = sys.executable

if IS_FROZEN:
    # dentro do .exe: PyInstaller descompacta em <pasta do exe>/_internal
    BUNDLE = Path(sys._MEIPASS) if hasattr(sys, "_MEIPASS") else \
        Path(sys.executable).resolve().parent / "_internal"
    APP_DIR = BUNDLE                      # contém gss/ e bin/
else:
    APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# os scripts de etapa rodam como subprocesso do MESMO binário congelado,
# então precisam saber onde está o pacote `gss`.
os.environ["GSS_APP_DIR"] = str(APP_DIR)

# ---------------------------------------------------------------- estado global
CFG: dict = {}
JOBS: dict = {"prepare": None, "train": None}
LOCK = threading.Lock()
GPU_CACHE: dict = {"checked": 0.0, "cuda": False, "name": ""}


def repo() -> Path:
    return Path(CFG["repo"])


def proj_paths() -> dict:
    root = Path(CFG["project"])
    return {
        "root": root,
        "repo": repo(),
        "wavs": root / "corpus" / "wavs",
        "metadata": root / "corpus" / "metadata.csv",
        "textgrids": root / "corpus" / "TextGrid" / "LJSpeech",
        "preprocessed": root / "corpus" / "preprocessed_data" / "LJSpeech",
        "config": root / "preprocess.yaml",
        "stats": root / "corpus" / "preprocessed_data" / "LJSpeech" / "stats.json",
        "logs": root / "logs",
        "outputs": root / "outputs",
        "state": root / "project.json",
        "official": repo() / "checkpoints" / "grainspeech_l1_ssim_gvar.ckpt",
        "vocoder": repo() / "hifigan" / "LJ_V2" / "generator_v2",
        "repo_config": repo() / "configs" / "LJSpeech" / "preprocess.yaml",
    }


def read_state() -> dict:
    p = proj_paths()["state"]
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_state(**kw) -> None:
    p = proj_paths()["state"]
    st = read_state()
    st.update(kw)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(st, indent=2, ensure_ascii=False), encoding="utf-8")


def tail_log(path: Path, n: int = 60) -> str:
    path = Path(path)
    if not path.is_file():
        return ""
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    data = data.replace("\r", "\n")
    keep = []
    for ln in data.splitlines():
        ln = ln.rstrip()
        if not ln.strip():
            continue
        # barra de progresso do Lightning: fica só a última versão da linha
        if re.match(r"^Epoch \d+", ln):
            keep = [k for k in keep if not re.match(r"^Epoch \d+", k)]
        keep.append(ln)
    return "\n".join(keep[-n:])


def running(name: str) -> bool:
    j = JOBS.get(name)
    return bool(j and j["proc"] and j["proc"].poll() is None)


def start_job(name: str, cmd: list[str], log_path: Path, extra_env: dict | None = None):
    with LOCK:
        if running(name):
            return False, "já existe uma tarefa em andamento"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.setdefault("OMP_NUM_THREADS", "1")
        pp = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
        for cand in (str(APP_DIR), str(repo() / "grainspeech")):
            if cand not in pp:
                pp.append(cand)
        env["PYTHONPATH"] = os.pathsep.join(pp)
        env["GSS_APP_DIR"] = str(APP_DIR)
        env.setdefault("GSS_NLTK_DIR", str(Path(CFG["project"]) / "nltk_data"))
        env["NLTK_DATA"] = env["GSS_NLTK_DIR"]
        if IS_FROZEN:
            env["GSS_FROZEN_PY"] = str(Path(sys.executable))
        if extra_env:
            env.update(extra_env)
        fh = open(log_path, "w", encoding="utf-8", errors="replace")
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=env, cwd=str(Path.cwd()))
        JOBS[name] = {"proc": proc, "log": str(log_path), "fh": fh,
                      "started": time.time(), "cmd": cmd}
        return True, " ".join(str(c) for c in cmd)


def finish_job(name: str) -> int | None:
    j = JOBS.get(name)
    if not j or j["proc"].poll() is None:
        return None
    rc = j["proc"].returncode
    try:
        j["fh"].close()
    except Exception:
        pass
    return rc


def gpu_info() -> dict:
    now = time.time()
    if now - GPU_CACHE["checked"] < 60:
        return GPU_CACHE
    GPU_CACHE.update(checked=now, cuda=False, name="")
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
        if out.returncode == 0 and out.stdout.strip():
            GPU_CACHE.update(cuda=True, name=out.stdout.strip().splitlines()[0])
            return GPU_CACHE
    except Exception:
        pass
    try:
        out = subprocess.run([PY, "-c", "import torch;print(int(torch.cuda.is_available()))"],
                             capture_output=True, text=True, timeout=90)
        GPU_CACHE.update(cuda=out.stdout.strip().endswith("1"))
    except Exception:
        pass
    return GPU_CACHE


def which_mfa() -> str:
    if CFG.get("mfa_bin"):
        return CFG["mfa_bin"]
    env_bin = os.environ.get("GSS_MFA_BIN")
    if env_bin and Path(env_bin).exists():
        return env_bin
    found = shutil.which("mfa")
    if found:
        return found
    for root in (Path(os.environ.get("MAMBA_ROOT_PREFIX", "/nonexistent")),
                 Path(os.environ.get("CONDA_PREFIX", "/nonexistent")),
                 Path("C:/ProgramData/miniconda3"), Path("C:/ProgramData/mambaforge")):
        for envs in list((root / "envs").glob("*")) + [root]:
            for nm in ("bin/mfa", "Scripts/mfa.exe", "mfa.bat"):
                if (envs / nm).exists():
                    return str(envs / nm)
    return ""


def list_runs() -> list[dict]:
    logs = proj_paths()["logs"]
    if not logs.is_dir():
        return []
    out = []
    for d in sorted(logs.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        summ = d / "train_summary.json"
        ck = sorted((d / "checkpoints").glob("*.ckpt")) if (d / "checkpoints").is_dir() else []
        metrics = d / "metrics.jsonl"
        curve = []
        if metrics.is_file():
            for ln in metrics.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    curve.append(json.loads(ln))
                except Exception:
                    pass
        info = {"name": d.name, "checkpoints": len(ck),
                "last_ckpt": str(d / "checkpoints" / "last.ckpt")
                if (d / "checkpoints" / "last.ckpt").exists() else "",
                "curve": curve[-400:], "mtime": d.stat().st_mtime}
        if summ.is_file():
            try:
                info["summary"] = json.loads(summ.read_text(encoding="utf-8"))
            except Exception:
                info["summary"] = {}
        if ck:
            losses = []
            for c in ck:
                m = re.search(r"loss([\-0-9.]+)", c.name)
                if m:
                    val_str = m.group(1).rstrip(".")
                    try:
                        losses.append((float(val_str), c.name))
                    except ValueError:
                        pass
            if losses:
                info["best_loss"] = min(losses)[0]
        out.append(info)
    return out


# ---------------------------------------------------------------- ações
def do_prepare(body: dict) -> dict:
    p = proj_paths()
    audio = Path(body.get("audio_dir") or "").expanduser()
    if not audio.is_dir():
        return {"ok": False, "error": f"pasta de áudio não existe: {audio}"}
    cmd = [PY, str(APP_DIR / "bin" / "gss_prepare.py"),
           "--root", str(p["root"]), "--repo", str(p["repo"]),
           "--audio-dir", str(audio),
           "--aligner", body.get("aligner", "mfa"),
           "--whisper-size", body.get("whisper_size", "base.en"),
           "--language", body.get("language", "en"),
           "--jobs", str(body.get("jobs", 1)),
           "--val-size", str(body.get("val_size", 6))]
    if CFG.get("mfa_bin"):
        cmd += ["--mfa-bin", CFG["mfa_bin"]]
    if CFG.get("mfa_root"):
        cmd += ["--mfa-root", CFG["mfa_root"]]
    ok, msg = start_job("prepare", cmd, p["logs"] / "prepare.log")
    return {"ok": ok, "msg": msg}


def do_train(body: dict) -> dict:
    p = proj_paths()
    if not (p["preprocessed"] / "train.txt").is_file():
        return {"ok": False, "error": "dataset não preparado — rode a etapa 1"}
    mode = body.get("mode", "fast")
    run = body.get("run_name") or f"{mode}_{time.strftime('%m%d_%H%M%S')}"
    cmd = [PY, str(APP_DIR / "bin" / "gss_train.py"),
           "--repo", str(p["repo"]),
           "--preprocess-config", str(p["config"]),
           "--run-name", run, "--mode", mode,
           "--batch-size", str(body.get("batch_size", 8)),
           "--num-workers", "0", "--threads", str(body.get("threads", 1)),
           "--log-dir", str(p["logs"])]
    if body.get("epochs"):
        cmd += ["--max-epochs", str(body["epochs"])]
    if body.get("lr"):
        cmd += ["--lr", str(body["lr"])]
    if not gpu_info()["cuda"]:
        cmd += ["--precision", "32-true"]
    if body.get("no_val_audio"):
        cmd += ["--no-val-audio"]
    ok, msg = start_job("train", cmd, p["logs"] / f"train_{run}.log")
    if ok:
        save_state(last_run=run, last_mode=mode)
        JOBS["train"]["run"] = run
    return {"ok": ok, "msg": msg, "run": run}


def do_infer(body: dict) -> dict:
    p = proj_paths()
    text = (body.get("text") or "").strip()
    if not text:
        return {"ok": False, "error": "texto vazio"}
    which = body.get("checkpoint", "official")
    if which == "official":
        ck, stats = p["official"], repo() / "configs" / "LJSpeech" / "stats.json"
    else:
        ck = Path(which)
        stats = p["stats"] if p["stats"].is_file() else None
    if not ck.is_file():
        return {"ok": False, "error": f"checkpoint não encontrado: {ck}"}
    p["outputs"].mkdir(parents=True, exist_ok=True)
    name = body.get("name") or f"audio_{int(time.time())}"
    name = re.sub(r"[^\w\-.]+", "_", name)[:60]
    out = p["outputs"] / f"{name}.wav"
    cmd = [PY, str(APP_DIR / "bin" / "gss_infer.py"),
           "--repo", str(p["repo"]), "--checkpoint", str(ck),
           "--text", text, "--output", str(out),
           "--device", body.get("device", "auto")]
    if stats:
        cmd += ["--stats", str(stats)]
    else:
        cmd += ["--preprocess-config", str(p["repo_config"])]
    t0 = time.time()
    env = dict(os.environ)
    pp = [x for x in env.get("PYTHONPATH", "").split(os.pathsep) if x]
    for cand in (str(APP_DIR), str(repo() / "grainspeech")):
        if cand not in pp:
            pp.append(cand)
    env["PYTHONPATH"] = os.pathsep.join(pp)
    env["GSS_APP_DIR"] = str(APP_DIR)
    env["GSS_NLTK_DIR"] = str(Path(CFG["project"]) / "nltk_data")
    env["NLTK_DATA"] = env["GSS_NLTK_DIR"]
    logp = p["logs"] / "infer.log"
    logp.parent.mkdir(parents=True, exist_ok=True)
    with logp.open("w", encoding="utf-8", errors="replace") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    dt = time.time() - t0
    if rc != 0 or not out.is_file():
        return {"ok": False, "error": tail_log(logp, 25), "seconds": round(dt, 1)}
    return {"ok": True, "wav": f"/audio/{out.name}", "seconds": round(dt, 1),
            "bytes": out.stat().st_size, "log": tail_log(logp, 8)}


def do_stop(body: dict) -> dict:
    name = body.get("name", "train")
    j = JOBS.get(name)
    if j and j["proc"].poll() is None:
        j["proc"].terminate()
        try:
            j["proc"].wait(timeout=20)
        except Exception:
            j["proc"].kill()
        return {"ok": True, "msg": f"{name} interrompido"}
    return {"ok": False, "msg": "nada para interromper"}


def parse_multipart_data(raw_bytes: bytes, content_type: str) -> tuple[dict, dict]:
    fields = {}
    files = {}
    m = re.search(r"boundary=(.+)", content_type)
    if not m:
        return fields, files
    boundary = m.group(1).strip().strip('"').encode("ascii")
    parts = raw_bytes.split(b"--" + boundary)
    for part in parts:
        if not part or part == b"--\r\n" or part == b"--":
            continue
        if b"\r\n\r\n" not in part:
            continue
        header_bytes, body_bytes = part.split(b"\r\n\r\n", 1)
        if body_bytes.endswith(b"\r\n"):
            body_bytes = body_bytes[:-2]
        headers_str = header_bytes.decode("utf-8", "ignore")
        fn_match = re.search(r'content-disposition:.*name="([^"]+)";\s*filename="([^"]+)"', headers_str, re.IGNORECASE)
        if fn_match:
            field_name = fn_match.group(1)
            filename = fn_match.group(2)
            files[field_name] = {"filename": filename, "data": body_bytes}
            continue
        name_match = re.search(r'content-disposition:.*name="([^"]+)"', headers_str, re.IGNORECASE)
        if name_match:
            field_name = name_match.group(1)
            fields[field_name] = body_bytes.decode("utf-8", "ignore").strip()
    return fields, files


def do_mp3_train(body: dict) -> dict:
    p = proj_paths()
    upload_dir = APP_DIR.parent / "audio_upload"
    upload_dir.mkdir(exist_ok=True)

    mp3_path = body.get("mp3_path", "")
    if not mp3_path or not Path(mp3_path).exists():
        file_name = body.get("file_name", "")
        if file_name:
            cand = upload_dir / file_name
            if cand.exists():
                mp3_path = str(cand)

    if not mp3_path or not Path(mp3_path).exists():
        file_name_raw = body.get("mp3_path", "").split("/")[-1].split("\\")[-1]
        file_name_raw = file_name_raw.replace(" ", "_")
        search_dirs = [
            upload_dir,
            APP_DIR.parent / "audios",
            APP_DIR.parent,
            ROOT / "datasets",
            ROOT / "audios",
        ]
        found_path = None
        for dir_path in search_dirs:
            if dir_path.exists() and (dir_path / file_name_raw).exists():
                found_path = str(dir_path / file_name_raw)
                break
        if not found_path:
            original_name = body.get("mp3_path", "").split("/")[-1].split("\\")[-1]
            for dir_path in search_dirs:
                if dir_path.exists() and (dir_path / original_name).exists():
                    found_path = str(dir_path / original_name)
                    break
        if found_path:
            mp3_path = found_path
        else:
            return {"ok": False, "error": f"❌ Arquivo MP3 não encontrado. Por favor selecione o arquivo MP3 no formulário."}

    run_name = body.get("run_name", "mp3_finetune")
    mode = body.get("mode", "fast")
    epochs = int(body.get("epochs", 150))
    language = body.get("language", "pt")

    cmd = [
        PY, str(APP_DIR / "finetune_mp3.py"),
        "--mp3", mp3_path,
        "--project", str(p["root"]),
        "--repo", str(p["repo"]),
        "--run-name", run_name,
        "--mode", mode,
        "--epochs", str(epochs),
        "--language", language
    ]

    ok, msg = start_job("mp3_train", cmd, p["logs"] / f"mp3_train_{run_name}.log")
    return {"ok": ok, "msg": msg, "run": run_name, "file": str(mp3_path)}


def full_status() -> dict:
    for k in ("prepare", "train"):
        finish_job(k)
    p = proj_paths()
    st = read_state()
    env = {
        "python": sys.version.split()[0],
        "frozen": IS_FROZEN,
        "gpu": gpu_info(),
        "mfa": which_mfa(),
        "repo_ok": (p["official"]).is_file() and (p["vocoder"]).is_file(),
        "repo": str(p["repo"]),
        "project": str(p["root"]),
    }
    jobs = {}
    for k, j in JOBS.items():
        if not j:
            jobs[k] = {"running": False}
            continue
        alive = j["proc"].poll() is None
        jobs[k] = {"running": alive, "rc": j["proc"].returncode,
                   "elapsed": round(time.time() - j["started"], 1),
                   "run": j.get("run", ""),
                   "log": tail_log(Path(j["log"]), 40)}
    n_wavs = len(list(p["wavs"].glob("*.wav"))) if p["wavs"].is_dir() else 0
    n_tg = len(list(p["textgrids"].glob("*.TextGrid"))) if p["textgrids"].is_dir() else 0
    n_tr = (len((p["preprocessed"] / "train.txt").read_text().splitlines())
            if (p["preprocessed"] / "train.txt").is_file() else 0)
    runs = list_runs()
    ckopts = [{"id": "official", "label": "Oficial (LJSpeech, sem treino)"}]
    for r in runs:
        if r["last_ckpt"]:
            loss = r.get("best_loss")
            lab = f"{r['name']}" + (f"  —  loss {loss:.3f}" if loss is not None else "")
            ckopts.append({"id": r["last_ckpt"], "label": lab})
    wavs_out = []
    if p["outputs"].is_dir():
        for f in sorted(p["outputs"].glob("*.wav"), key=lambda x: x.stat().st_mtime,
                        reverse=True)[:20]:
            wavs_out.append({"name": f.name, "url": f"/audio/{f.name}",
                             "kb": round(f.stat().st_size / 1024, 1)})
    return {"env": env, "jobs": jobs, "state": st,
            "dataset": {"wavs": n_wavs, "textgrids": n_tg, "train": n_tr,
                        "prepared": bool(st.get("prepared")),
                        "minutes": st.get("minutes", 0),
                        "aligner": st.get("aligner", ""),
                        "has_stats": p["stats"].is_file()},
            "runs": runs, "checkpoint_options": ckopts, "outputs": wavs_out}


# ---------------------------------------------------------------- HTML
PAGE = r"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GrainSpeech Studio</title>
<style>
:root{--bg:#0e1116;--panel:#161b22;--panel2:#1c2330;--bd:#2b3441;--fg:#e6edf3;
--mut:#8b98a9;--acc:#4ea1ff;--ok:#3fb950;--warn:#d29922;--err:#f85149}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 ui-sans-serif,system-ui,"Segoe UI",Roboto,Arial,sans-serif}
header{padding:18px 22px;border-bottom:1px solid var(--bd);background:var(--panel);
display:flex;gap:16px;align-items:center;flex-wrap:wrap}
header h1{font-size:17px;margin:0;font-weight:650;letter-spacing:.2px}
header .sub{color:var(--mut);font-size:12px}
.pill{padding:3px 9px;border-radius:999px;font-size:11.5px;border:1px solid var(--bd);
background:var(--panel2);color:var(--mut);white-space:nowrap}
.pill.ok{color:var(--ok);border-color:#1f4429}.pill.err{color:var(--err);border-color:#4d2226}
.pill.warn{color:var(--warn);border-color:#4d3d1c}
main{max-width:1180px;margin:0 auto;padding:20px;display:grid;gap:16px}
.card{background:var(--panel);border:1px solid var(--bd);border-radius:12px;overflow:hidden}
.card>h2{margin:0;padding:13px 16px;font-size:13.5px;font-weight:650;background:var(--panel2);
border-bottom:1px solid var(--bd);display:flex;align-items:center;gap:10px}
.card>h2 .n{display:inline-grid;place-items:center;width:21px;height:21px;border-radius:50%;
background:var(--acc);color:#04101f;font-size:11.5px;font-weight:800}
.body{padding:16px;display:grid;gap:12px}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
label{display:grid;gap:5px;font-size:12px;color:var(--mut)}
input,select,textarea{background:#0b0f14;border:1px solid var(--bd);color:var(--fg);
border-radius:8px;padding:8px 10px;font:inherit;font-size:13px}
input:focus,select:focus,textarea:focus{outline:none;border-color:var(--acc)}
textarea{width:100%;min-height:74px;resize:vertical}
button{background:var(--acc);color:#04101f;border:0;border-radius:8px;padding:9px 15px;
font:inherit;font-size:13px;font-weight:650;cursor:pointer}
button:hover{filter:brightness(1.09)}
button:disabled{background:#2a3441;color:#7b8798;cursor:not-allowed}
button.ghost{background:transparent;color:var(--fg);border:1px solid var(--bd)}
button.danger{background:#3a1d21;color:#ffb3ad;border:1px solid #5c2b30}
pre{margin:0;background:#0b0f14;border:1px solid var(--bd);border-radius:8px;padding:11px;
font:11.5px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace;color:#b9c6d6;
max-height:260px;overflow:auto;white-space:pre-wrap;word-break:break-word}
table{width:100%;border-collapse:collapse;font-size:12.5px}
td,th{padding:6px 9px;border-bottom:1px solid var(--bd);text-align:left}
th{color:var(--mut);font-weight:600;font-size:11.5px;text-transform:uppercase;letter-spacing:.4px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media(max-width:820px){.grid2{grid-template-columns:1fr}}
.hint{color:var(--mut);font-size:12px}
.big{font-size:20px;font-weight:700}
audio{width:100%;height:38px;margin-top:6px}
.wlist{display:grid;gap:8px}
.witem{border:1px solid var(--bd);border-radius:9px;padding:9px 11px;background:var(--panel2)}
canvas{width:100%;height:150px;background:#0b0f14;border:1px solid var(--bd);border-radius:8px}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px}
.tab{padding:6px 12px;border:1px solid var(--bd);border-radius:999px;background:var(--panel2);
color:var(--mut);font-size:12px;cursor:pointer}
.tab.on{color:#04101f;background:var(--acc);border-color:var(--acc);font-weight:650}
.spin{display:inline-block;width:12px;height:12px;border:2px solid var(--mut);
border-top-color:var(--acc);border-radius:50%;animation:s .8s linear infinite}
@keyframes s{to{transform:rotate(360deg)}}
.hide{display:none}
</style></head><body>
<header>
  <div><h1>GrainSpeech Studio</h1>
  <div class="sub">áudio &rarr; Whisper &rarr; alinhamento forçado &rarr; treino &rarr; TTS</div></div>
  <span class="pill" id="pRepo">repo…</span>
  <span class="pill" id="pGpu">gpu…</span>
  <span class="pill" id="pMfa">mfa…</span>
  <span class="pill" id="pDs">dataset…</span>
</header>
<main>

<section class="card"><h2><span class="n">1</span> Preparar dataset
  <span class="hint" id="prepState"></span></h2>
 <div class="body">
  <div class="row">
   <label style="flex:1;min-width:280px">Pasta com seus áudios (.wav/.mp3/.flac)
    <input id="audioDir" placeholder="C:\meus_audios" style="width:100%"></label>
   <label>Transcrição
    <select id="trMode"><option value="auto">auto (metadata.csv/.lab/.txt, senão Whisper)</option>
    <option value="whisper">forçar Whisper</option></select></label>
   <label>Modelo Whisper
    <select id="whisperSize"><option value="tiny.en">tiny.en (rápido)</option>
    <option value="base.en" selected>base.en (recomendado)</option>
    <option value="small.en">small.en (melhor)</option>
    <option value="base">base (multilíngue)</option>
    <option value="small">small (multilíngue)</option></select></label>
   <label>Idioma<input id="lang" value="en" style="width:70px"></label>
  </div>
  <div class="row">
   <label>Alinhador
    <select id="aligner"><option value="mfa" selected>MFA — alinhamento forçado real</option>
    <option value="uniform">uniforme (fallback sem MFA, baixa precisão)</option></select></label>
   <label>Jobs MFA<input id="jobs" type="number" value="1" min="1" max="8" style="width:70px"></label>
   <label>Tamanho validação<input id="valSize" type="number" value="6" min="1" style="width:80px"></label>
   <button id="btnPrep">Preparar dataset</button>
   <button class="danger hide" id="btnStopPrep">Parar</button>
  </div>
  <div class="hint">Já existe <code>metadata.csv</code> ou <code>.lab</code>/<code>.txt</code>
   ao lado dos áudios? Eles são usados e o Whisper é pulado.</div>
  <pre id="prepLog" class="hide"></pre>
 </div></section>

<section class="card"><h2><span class="n">2</span> Treinar
  <span class="hint" id="trainState"></span></h2>
 <div class="body">
  <div class="tabs">
   <div class="tab on" data-m="fast">⚡ Rápido — fine-tune do modelo oficial</div>
   <div class="tab" data-m="scratch">🐢 Normal — do zero (estilo paper)</div>
  </div>
  <div class="hint" id="modeHint"></div>
  <div class="row">
   <label>Épocas<input id="epochs" type="number" value="150" min="1" style="width:100px"></label>
   <label>Batch<input id="bs" type="number" value="8" min="1" style="width:80px"></label>
   <label>Learning rate<input id="lr" type="text" placeholder="auto" style="width:100px"></label>
   <label>Threads<input id="threads" type="number" value="2" min="1" style="width:70px"></label>
   <label style="align-self:end"><span>&nbsp;</span>
    <span><input type="checkbox" id="noVal"> sem vocoder na validação (economiza RAM)</span></label>
  </div>
  <div class="row">
   <label style="flex:1;min-width:200px">Nome do treino<input id="runName" placeholder="auto"></label>
   <button id="btnTrain">Treinar</button>
   <button class="danger hide" id="btnStopTrain">Parar</button>
  </div>
  <canvas id="chart" height="150"></canvas>
  <pre id="trainLog" class="hide"></pre>
 </div></section>

<section class="card"><h2><span class="n">3</span> Gerar áudio</h2>
 <div class="body">
  <textarea id="text" placeholder="Digite o texto em inglês…">Small models can give every word a voice.</textarea>
  <div class="row">
   <label style="flex:1;min-width:260px">Checkpoint
    <select id="ckpt"></select></label>
   <label>Device<select id="device"><option value="auto">auto</option>
    <option value="cpu">cpu</option><option value="cuda">cuda</option></select></label>
   <button id="btnGen">Gerar áudio</button>
  </div>
  <div id="genOut"></div>
  <div class="hint">Histórico:</div>
  <div class="wlist" id="outs"></div>
 </div></section>

<section class="card"><h2><span class="n">⚡</span> Upload MP3 + Treino Rápido
  <span class="hint">upload .mp3 → transcreve → treina automaticamente</span></h2>
 <div class="body">
  <div class="row">
   <label style="flex:1;min-width:260px">Arquivo .mp3
    <input type="file" id="mp3File" accept=".mp3,.wav,.flac,audio/*" style="width:100%"></label>
   <label>Nome do treino<input id="mp3RunName" value="mp3_finetune" style="width:140px"></label>
   <label>Épocas<input id="mp3Epochs" type="number" value="150" min="1" style="width:80px"></label>
  </div>
  <div class="row">
   <label>Modo
    <select id="mp3Mode"><option value="fast" selected>⚡ Rápido (fine-tune)</option>
    <option value="scratch">🐢 Normal (do zero)</option></select></label>
   <button id="btnMp3Train">📤 Upload + Treinar</button>
  </div>
  <div class="hint">Não precisa criar dataset manualmente. O arquivo é copiado, transcrito (Whisper/nltk) e o treino inicia automaticamente.</div>
  <pre id="mp3Log" class="hide"></pre>
  <div style="margin-top:8px;background:#0b0f14;border:1px solid var(--bd);border-radius:8px;padding:10px;">
   <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;font-size:13px;color:var(--mut);">
    <span style="font-weight:700;color:var(--fg);">⏱ Progresso do Treino MP3</span>
    <span style="margin-left:auto;font-size:11px;color:#4ea1ff;">Atualiza automaticamente a cada 3s</span>
   </div>
   <div style="width:100%;height:12px;background:#161b22;border:1px solid #2b3441;border-radius:6px;overflow:hidden;position:relative;">
    <div id="mp3ProgressLive" style="width:0%;height:100%;background:linear-gradient(90deg,#d29922,#4ea1ff);transition:width .6s ease;"></div>
   </div>
   <div id="mp3StatusLive" style="font-size:12px;color:#b9c6d6;margin-top:6px;font-family:monospace;min-height:40px;padding:4px;background:#0b0f14;border-radius:4px;">
    Aguardando início... Selecione um arquivo .mp3 e clique "📤 Upload + Treinar"
   </div>
  </div>
 </div></section>

<section class="card"><h2><span class="n">i</span> Treinos e ambiente</h2>
 <div class="body"><div id="runs"></div><div id="envBox" class="hint"></div></div></section>

</main>
<script>
let S=null,MODE='fast';
const $=id=>document.getElementById(id);
const H=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function pill(el,txt,cls){el.className='pill'+(cls?' '+cls:'');el.textContent=txt;}

async function api(path,body){
 const o=body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{};
 const r=await fetch(path,o);return r.json();
}
function fmtSec(s){if(s==null)return'';s=Math.round(s);const m=Math.floor(s/60);
 return m?m+'m '+(s%60)+'s':s+'s';}

function draw(runs){
 const c=$('chart'),x=c.getContext('2d');
 const W=c.width=c.clientWidth*2,h=c.height=300;x.scale(1,1);
 x.clearRect(0,0,W,h);
 const cur=(runs.find(r=>r.name===(S?.state?.last_run))||runs[0]||{}).curve||[];
 const pts=cur.filter(p=>p.loss!=null);
 if(pts.length<2){x.fillStyle='#8b98a9';x.font='22px sans-serif';
  x.fillText('sem dados de treino ainda',20,40);return;}
 const xs=pts.map(p=>p.epoch),ys=pts.map(p=>p.loss);
 const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
 const px=e=>20+(e-x0)/Math.max(1,(x1-x0))*(W-40);
 const py=v=>h-24-(v-y0)/Math.max(1e-6,(y1-y0))*(h-48);
 x.strokeStyle='#2b3441';x.lineWidth=1;
 for(let i=0;i<=4;i++){const y=24+i*(h-48)/4;x.beginPath();x.moveTo(20,y);x.lineTo(W-20,y);x.stroke();
  x.fillStyle='#8b98a9';x.font='18px sans-serif';
  x.fillText((y1-(y1-y0)*i/4).toFixed(2),W-16,y+6);}
 x.strokeStyle='#4ea1ff';x.lineWidth=3;x.beginPath();
 pts.forEach((p,i)=>{i?x.lineTo(px(p.epoch),py(p.loss)):x.moveTo(px(p.epoch),py(p.loss));});
 x.stroke();
 const v=cur.filter(p=>p.val_loss!=null);
 if(v.length){x.fillStyle='#3fb950';v.forEach(p=>{x.beginPath();
  x.arc(px(p.epoch),py(p.val_loss),5,0,7);x.fill();});}
 x.fillStyle='#e6edf3';x.font='20px sans-serif';
 x.fillText('loss (azul) / val_loss (verde) — época '+x0+'→'+x1,20,20);
}

function render(){
 if(!S)return;
 const e=S.env;
 pill($('pRepo'),e.repo_ok?'repo ok':'repo FALTANDO',e.repo_ok?'ok':'err');
 pill($('pGpu'),e.gpu.cuda?('GPU: '+e.gpu.name):'CPU',e.gpu.cuda?'ok':'warn');
 pill($('pMfa'),e.mfa?'MFA ok':'MFA ausente',e.mfa?'ok':'warn');
 const d=S.dataset;
 pill($('pDs'),d.prepared?`${d.wavs} áudios · ${d.minutes} min · ${d.train} treino`:'sem dataset',
  d.prepared?'ok':'');
 $('prepState').innerHTML=d.prepared?
  `— <b>${d.wavs}</b> utterances, <b>${d.minutes}</b> min, alinhador <b>${H(d.aligner)}</b>,
   ${d.textgrids} TextGrids, ${d.train} treino/${S.state.n_val??'?'} val`:'';
 const jp=S.jobs.prepare||{},jt=S.jobs.train||{};
 $('btnPrep').disabled=jp.running;
 $('btnStopPrep').classList.toggle('hide',!jp.running);
 if(jp.running){$('prepLog').classList.remove('hide');$('prepLog').textContent=jp.log||'aguarde…';}
 else if(jp.rc!=null){$('prepLog').classList.remove('hide');
  $('prepLog').textContent=(jp.rc===0?'✅ concluído\n':'❌ falhou (rc='+jp.rc+')\n')+(jp.log||'');}
 $('btnTrain').disabled=jt.running||!d.prepared;
 $('btnStopTrain').classList.toggle('hide',!jt.running);
 $('trainState').innerHTML=jt.running?
  `<span class="spin"></span> treinando ${H(jt.run||'')} — ${fmtSec(jt.elapsed)}`:
  (jt.rc!=null?(jt.rc===0?'— concluído ✅':'— falhou ❌'):'');
 if(jt.running||jt.rc!=null){$('trainLog').classList.remove('hide');
  $('trainLog').textContent=(jt.log||'aguarde…');}
 const sel=$('ckpt'),cur=sel.value;
 sel.innerHTML=S.checkpoint_options.map(o=>`<option value="${H(o.id)}">${H(o.label)}</option>`).join('');
 if([...sel.options].some(o=>o.value===cur))sel.value=cur;
 $('outs').innerHTML=S.outputs.length?S.outputs.map(w=>
  `<div class="witem"><b>${H(w.name)}</b> <span class="hint">${w.kb} kB</span>
   <audio controls preload="none" src="${H(w.url)}"></audio></div>`).join(''):
  '<span class="hint">nenhum áudio gerado ainda</span>';
 $('runs').innerHTML=S.runs.length?`<table><tr><th>treino</th><th>modo</th><th>épocas</th>
  <th>loss</th><th>tempo</th><th>s/época</th><th>ckpts</th></tr>`+S.runs.map(r=>{
   const s=r.summary||{};return `<tr><td><b>${H(r.name)}</b></td>
   <td>${H(s.mode||'')}</td><td>${s.max_epochs??''}</td>
   <td>${r.best_loss!=null?r.best_loss.toFixed(3):'—'}</td>
   <td>${fmtSec(s.seconds_per_epoch?(s.max_epochs*s.seconds_per_epoch):null)}</td>
   <td>${s.seconds_per_epoch??''}</td><td>${r.checkpoints}</td></tr>`;}).join('')+'</table>'
  :'<span class="hint">nenhum treino ainda</span>';
 $('envBox').innerHTML=`python ${H(e.python)} · projeto <code>${H(e.project)}</code> ·
  repo <code>${H(e.repo)}</code> · MFA <code>${H(e.mfa||'não encontrado')}</code> ·
  ${e.frozen?'empacotado (exe)':'modo script'}`;
 draw(S.runs);
}
const HINTS={fast:'⚡ Copia os 264.881 parâmetros acústicos do checkpoint oficial e ajusta no seu '
 +'dataset. É o caminho para <b>poucos dados</b>: com minutos de áudio já sai fala inteligível. '
 +'Sugestão: 100–300 épocas, lr 2e-4.',
 scratch:'🐢 Treino do zero, como no paper (LJSpeech, 5000 épocas, batch 128). Com poucos minutos '
 +'de áudio o resultado será ruído — use só se tiver horas de gravação e paciência.'};
function setMode(m){MODE=m;$('modeHint').innerHTML=HINTS[m];
 document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('on',t.dataset.m===m));
 $('epochs').value=m==='fast'?150:2000;$('bs').value=m==='fast'?8:16;
 $('lr').value=m==='fast'?'2e-4':'1e-3';}
document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>setMode(t.dataset.m));

$('btnPrep').onclick=async()=>{
 const r=await api('/api/prepare',{audio_dir:$('audioDir').value.trim(),
  aligner:$('aligner').value,whisper_size:$('whisperSize').value,
  language:$('lang').value.trim()||'en',jobs:+$('jobs').value||1,
  val_size:+$('valSize').value||6});
 if(!r.ok)alert('Erro: '+r.error);refresh();};
$('btnStopPrep').onclick=()=>api('/api/stop',{name:'prepare'});
$('btnTrain').onclick=async()=>{
 const r=await api('/api/train',{mode:MODE,run_name:$('runName').value.trim(),
  epochs:+$('epochs').value,batch_size:+$('bs').value,
  lr:$('lr').value.trim()?+$('lr').value.trim():null,
  threads:+$('threads').value||1,no_val_audio:$('noVal').checked});
 if(!r.ok)alert('Erro: '+r.error);refresh();};
$('btnStopTrain').onclick=()=>api('/api/stop',{name:'train'});
$('btnMp3Train').onclick=async()=>{
 $('btnMp3Train').disabled=true;$('mp3Log').classList.remove('hide');$('mp3Log').textContent='Enviando arquivo e iniciando treino...';
 const f=$('mp3File').files[0];
 if(!f){$('mp3Log').textContent='❌ Nenhum arquivo selecionado';$('btnMp3Train').disabled=false;return;}
 try{
   const fd=new FormData();
   fd.append('file', f);
   fd.append('run_name', $('mp3RunName').value.trim()||'mp3_finetune');
   fd.append('mode', $('mp3Mode').value||'fast');
   fd.append('epochs', $('mp3Epochs').value||'150');
   fd.append('language', $('mp3Lang')?$('mp3Lang').value:'pt');
   const res=await fetch('/api/mp3_train',{method:'POST',body:fd});
   const r=await res.json();
   if(!r.ok){$('mp3Log').textContent='❌ Erro: '+(r.error||'Falha no envio');}
   else{$('mp3Log').textContent='✅ Treino iniciado! Arquivo: '+(r.file||f.name)+'\n'+r.msg;}
 }catch(e){
   $('mp3Log').textContent='❌ Erro na requisição: '+e;
 }
 $('btnMp3Train').disabled=false;refresh();
};

$('btnGen').onclick=async()=>{
 $('btnGen').disabled=true;$('genOut').innerHTML='<span class="spin"></span> gerando…';
 const ck=$('ckpt').value;
 const r=await api('/api/infer',{text:$('text').value,checkpoint:ck,
  device:$('device').value,name:(MODE||'tts')+'_'+Date.now()});
 $('btnGen').disabled=false;
 $('genOut').innerHTML=r.ok?
  `<div class="witem"><b>gerado em ${r.seconds}s</b> · ${(r.bytes/1024).toFixed(0)} kB
   <audio controls autoplay src="${H(r.wav)}"></audio>
   <div class="hint"><code>${H((r.log||'').split('\n').filter(l=>l.includes('Phonemes')||l.includes('{')).pop()||'')}</code></div></div>`
  :`<span style="color:var(--err)">Erro: ${H(r.error||'')}</span>`;
 refresh();};
async function refresh(){S=await api('/api/state');render();}
setMode('fast');refresh();setInterval(refresh,2000);
</script></body></html>"""


# ---------------------------------------------------------------- servidor
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def log_message(self, *a):                      # silêncio no console
        pass

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif u.path == "/api/state":
            try:
                self._json(full_status())
            except Exception as exc:
                self._json({"error": str(exc)}, 500)
        elif u.path.startswith("/audio/"):
            name = Path(u.path.split("/audio/", 1)[1]).name
            f = proj_paths()["outputs"] / name
            if f.is_file():
                self._send(200, f.read_bytes(), "audio/wav",
                           {"Accept-Ranges": "bytes"})
            else:
                self._send(404, b"not found", "text/plain")
        elif u.path.startswith("/logs/"):
            name = Path(u.path.split("/logs/", 1)[1]).name
            self._json({"log": tail_log(proj_paths()["logs"] / name, 400)})
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n > 0 else b""
        content_type = self.headers.get("Content-Type", "")
        body = {}
        if "/api/mp3_train" in u.path and "multipart/form-data" in content_type:
            upload_dir = APP_DIR.parent / "audio_upload"
            upload_dir.mkdir(exist_ok=True)
            fields, files = parse_multipart_data(raw, content_type)
            body = fields
            if "file" in files:
                finfo = files["file"]
                clean_filename = Path(finfo["filename"]).name.replace(" ", "_")
                saved_path = upload_dir / clean_filename
                saved_path.write_bytes(finfo["data"])
                body["mp3_path"] = str(saved_path)
                body["file_name"] = clean_filename
        else:
            try:
                body = json.loads(raw or b"{}")
            except Exception:
                body = {}
        routes = {"/api/prepare": do_prepare, "/api/train": do_train,
                  "/api/infer": do_infer, "/api/stop": do_stop,
                  "/api/mp3_train": do_mp3_train}
        fn = routes.get(u.path)
        if not fn:
            return self._json({"error": "rota desconhecida"}, 404)
        try:
            self._json(fn(body))
        except Exception as exc:
            import traceback
            self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}\n"
                                              + traceback.format_exc()[-1200:]}, 500)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default=None,
                    help="pasta do projeto (default: ./GrainSpeechStudioProject)")
    ap.add_argument("--repo", default=None,
                    help="pasta do clone do GrainSpeech")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8756)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--mfa-bin", default=None)
    ap.add_argument("--mfa-root", default=None)
    args = ap.parse_args()

    base = Path(sys.argv[0]).resolve().parent if IS_FROZEN else Path.cwd()
    project = Path(args.project) if args.project else (base / "GrainSpeechStudioProject")
    repo_p = Path(args.repo) if args.repo else None
    if repo_p is None:
        for cand in (base / "GrainSpeech", Path.cwd() / "GrainSpeech",
                     Path.home() / "GrainSpeech"):
            if (cand / "checkpoints").is_dir():
                repo_p = cand
                break
    if repo_p is None:
        print("ERRO: não achei o repositório GrainSpeech. Use --repo <pasta>.\n"
              "  git clone https://github.com/lab-emi/GrainSpeech.git")
        return 2
    CFG.update(project=str(project.resolve()), repo=str(repo_p.resolve()),
               mfa_bin=args.mfa_bin, mfa_root=args.mfa_root)
    project.mkdir(parents=True, exist_ok=True)
    (project / "logs").mkdir(exist_ok=True)
    (project / "outputs").mkdir(exist_ok=True)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print("=" * 66)
    print("  GrainSpeech Studio")
    print(f"  interface : {url}")
    print(f"  projeto   : {CFG['project']}")
    print(f"  repo      : {CFG['repo']}")
    print(f"  MFA       : {which_mfa() or 'não encontrado (use o alinhador uniforme)'}")
    print("=" * 66)
    print("  Ctrl+C para sair")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nencerrando…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
