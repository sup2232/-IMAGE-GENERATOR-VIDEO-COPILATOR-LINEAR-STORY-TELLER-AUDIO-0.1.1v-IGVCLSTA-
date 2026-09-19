import os
import sys
import time
import json
import uuid
import subprocess
import urllib.request
import urllib.error
from pathlib import Path

PKG = Path(__file__).resolve().parent / "pkg" / "GrainSpeechStudio_pacote"
PORT = 8805
BASE = f"http://127.0.0.1:{PORT}"

def log(msg: str):
    print(f"[E2E TEST] {msg}", flush=True)

def http_get(path: str, timeout=10):
    url = BASE + path
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()

def http_post_multipart(endpoint: str, fields: dict, file_field: str, file_path: Path):
    boundary = uuid.uuid4().hex
    body = bytearray()
    for k, v in fields.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode("utf-8"))
        body.extend(f"{v}\r\n".encode("utf-8"))

    filename = file_path.name
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'.encode("utf-8"))
    body.extend(b"Content-Type: audio/mpeg\r\n\r\n")
    body.extend(file_path.read_bytes())
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(
        BASE + endpoint,
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.status, resp.read()

def http_post_json(endpoint: str, payload: dict):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE + endpoint,
        data=data,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.status, resp.read()

def main():
    log("Iniciando suíte de teste E2E...")

    # 1. Iniciar servidor gss_app em background
    server_cmd = [
        sys.executable,
        str(PKG / "app" / "gss_app.py"),
        "--project", str(PKG / "MeuProjeto"),
        "--repo", str(PKG / "GrainSpeech"),
        "--port", str(PORT),
        "--no-browser"
    ]
    log(f"Iniciando servidor na porta {PORT}...")
    server_log_path = Path(__file__).resolve().parent / "server.log"
    server_log = open(server_log_path, "w")
    srv_proc = subprocess.Popen(server_cmd, stdout=server_log, stderr=subprocess.STDOUT)

    try:
        # Aguarda servidor iniciar
        started = False
        for _ in range(30):
            time.sleep(0.5)
            try:
                st, _ = http_get("/")
                if st == 200:
                    started = True
                    break
            except Exception:
                pass

        if not started:
            raise RuntimeError("Servidor não iniciou a tempo")
        log("Servidor ativo e respondendo!")

        # 2. Testar upload de MP3 via multipart
        mp3_source = PKG / "audio_upload" / "musica_grind.mp3"
        if not mp3_source.exists():
            # Tenta encontrar qualquer mp3
            mp3s = list(PKG.rglob("*.mp3"))
            if not mp3s:
                raise FileNotFoundError("Nenhum MP3 de teste encontrado")
            mp3_source = mp3s[0]

        test_run_name = "teste_e2e"
        log(f"Enviando arquivo MP3 '{mp3_source.name}' para /api/mp3_train...")
        fields = {
            "run_name": test_run_name,
            "mode": "fast",
            "epochs": os.environ.get("T_EPOCHS", "3"),
            "language": "pt"
        }
        st, res_bytes = http_post_multipart("/api/mp3_train", fields, "file", mp3_source)
        res = json.loads(res_bytes)
        log(f"Resposta do upload: {res}")
        assert res.get("ok") is True, f"Upload falhou: {res}"

        # 3. Aguardar progresso do treino
        log("Monitorando execução do treino...")
        timeout_sec = int(os.environ.get("T_TIMEOUT", "600"))
        t0 = time.time()
        job_done = False
        while time.time() - t0 < timeout_sec:
            time.sleep(3)
            st, state_bytes = http_get("/api/state")
            if st != 200:
                log(f"Erro em /api/state: {st}")
                continue
            state = json.loads(state_bytes)
            job = state.get("jobs", {}).get("mp3_train", {})
            log_tail = (job.get("log") or "").splitlines()[-1:]
            last_line = log_tail[0] if log_tail else ""
            log(f"Running={job.get('running')}, rc={job.get('rc')} | {last_line[:80]}")

            if not job.get("running") and job.get("rc") is not None:
                if job.get("rc") == 0:
                    job_done = True
                else:
                    raise RuntimeError(f"Treino falhou com retcode {job.get('rc')}")
                break

        if not job_done:
            raise TimeoutError("Treino excedeu tempo limite no teste E2E")

        log("Treino concluído com sucesso!")

        # 4. Verificar se checkpoint aparece na API /api/state
        st, state_bytes = http_get("/api/state")
        state = json.loads(state_bytes)
        ckpts = state.get("checkpoint_options", [])
        e2e_ckpts = [c for c in ckpts if test_run_name in c.get("label", "")]
        log(f"Checkpoints para '{test_run_name}': {e2e_ckpts}")
        assert len(e2e_ckpts) > 0, "Nenhum checkpoint encontrado na lista de opções"
        chosen_ckpt = e2e_ckpts[0]["id"]

        # 5. Testar inferência TTS
        log(f"Testando geração de áudio com checkpoint '{chosen_ckpt}'...")
        infer_payload = {
            "text": "Olá, este é um teste completo do GrainSpeech Studio.",
            "checkpoint": chosen_ckpt,
            "device": "cpu",
            "name": "e2e_gen"
        }
        st, infer_bytes = http_post_json("/api/infer", infer_payload)
        infer_res = json.loads(infer_bytes)
        log(f"Resposta inferência: {infer_res}")
        assert infer_res.get("ok") is True, f"Inferência falhou: {infer_res}"

        wav_url = infer_res.get("wav")
        st, wav_data = http_get(wav_url)
        assert st == 200 and len(wav_data) > 1000, "Download do WAV falhou ou arquivo é muito pequeno"

        out_wav = PKG / "MeuProjeto" / "outputs" / "e2e_gen.wav"
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        out_wav.write_bytes(wav_data)
        log(f"SUCESSO TOTAL! Áudio gerado salvo em: {out_wav} ({len(wav_data)} bytes)")

    finally:
        srv_proc.terminate()
        srv_proc.wait(timeout=5)
        server_log.close()

if __name__ == "__main__":
    main()
