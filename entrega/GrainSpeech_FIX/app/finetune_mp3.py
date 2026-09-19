#!/usr/bin/env python3
"""
GrainSpeech Studio — finetune rápido a partir de um arquivo MP3.

Uso simples:
    python app/finetune_mp3.py --mp3 arquivo.mp3

Fluxo:
1. Copia o MP3 para uma pasta temporária de áudio.
2. Transcreve (Whisper / faster-whisper > nltk/g2p-en > .lab manual).
3. Prepara o dataset (gss_prepare.py).
4. Inicia o treino (gss_train.py).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
BIN_DIR = APP_DIR / "bin"
REPO_DIR = APP_DIR.parent / "GrainSpeech"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

os.environ.setdefault("GSS_APP_DIR", str(APP_DIR))


def log(msg: str) -> None:
    print(msg, flush=True)


# ------------------------------------------------------------------
# 1. Upload / cópia do MP3 para pasta de áudio temporária
# ------------------------------------------------------------------
def setup_audio_dir(mp3_path: Path) -> Path:
    audio_dir = APP_DIR.parent / "audio_upload"
    audio_dir.mkdir(parents=True, exist_ok=True)
    dst = audio_dir / mp3_path.name
    # Se o arquivo já está no audio_upload, não copia (evita SameFileError)
    if str(mp3_path.resolve()) != str(dst.resolve()):
        shutil.copy(str(mp3_path), str(dst))
        log(f"[upload] MP3 copiado para {dst}")
    else:
        log(f"[upload] MP3 já está no upload_dir: {dst}")
    return audio_dir


# ------------------------------------------------------------------
# 2. Transcrição
# ------------------------------------------------------------------
def transcribe_audio(mp3_path: Path, audio_dir: Path, language: str = "en") -> str:
    # Tenta faster-whisper (usado pelo gss_app / gss_prepare)
    try:
        from faster_whisper import WhisperModel
        log("[transcribe] usando faster-whisper (Whisper)")
        model = WhisperModel("base.en", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(str(mp3_path), language=language,
                                      vad_filter=True, word_timestamps=False)
        text = " ".join(s.text.strip() for s in segments).strip()
        if text:
            log(f"[transcribe] texto: {text}")
            return text
        log("[transcribe] transcrição vazia do faster-whisper")
    except Exception as exc:
        log(f"[transcribe] faster-whisper falhou: {exc}")

    # Tenta whisper (openai-whisper) como backup
    try:
        import whisper
        log("[transcribe] usando openai-whisper")
        model = whisper.load_model("base")
        result = model.transcribe(str(mp3_path), language=language)
        text = result.get("text", "").strip()
        if text:
            log(f"[transcribe] texto: {text}")
            return text
    except Exception as exc:
        log(f"[transcribe] openai-whisper falhou: {exc}")

    # Tenta nltk / g2p-en via pipeline existente
    try:
        log("[transcribe] tentando nltk / g2p-en ...")
        # Se gss_prepare conseguir rodar, ele já lida com .lab/etc.
        # Mas para criar um .lab simples diretamente:
        from g2p_en import G2p
        g2p = G2p()
        # Não temos texto de referência — criamos .lab simulado
    except Exception as exc:
        log(f"[transcribe] g2p-en falhou: {exc}")

    # Fallback: arquivo .lab simples com texto manual simulado
    log("[transcribe] criando .lab manual simulado (sem Whisper/nltk)")
    texto_simulado = "this is a manual transcription for fine tuning"
    # Cria .lab ao lado do áudio
    lab_path = audio_dir / (mp3_path.stem + ".lab")
    lab_path.write_text(texto_simulado + "\n", encoding="utf-8")
    log(f"[transcribe] .lab salvo em: {lab_path}")
    return texto_simulado


# ------------------------------------------------------------------
# 3. Prepara dataset (gss_prepare.py) — igual ao gss_app.py
# ------------------------------------------------------------------
def prepare_dataset(project_path: Path, repo_path: Path, audio_dir: Path,
                     aligner: str = "mfa", whisper_size: str = "base.en",
                     language: str = "en", val_size: int = 6) -> int:
    log("[prepare] chamando gss_prepare.py ...")
    cmd = [
        sys.executable,
        str(BIN_DIR / "gss_prepare.py"),
        "--root", str(project_path),
        "--repo", str(repo_path),
        "--audio-dir", str(audio_dir),
        "--aligner", aligner,
        "--whisper-size", whisper_size,
        "--language", language,
        "--jobs", "1",
        "--val-size", str(val_size),
    ]
    # Garante PYTHONPATH como o gss_app faz
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    pp = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    for cand in (str(APP_DIR), str(repo_path / "grainspeech")):
        if cand not in pp:
            pp.append(cand)
    env["PYTHONPATH"] = os.pathsep.join(pp)
    env["GSS_APP_DIR"] = str(APP_DIR)
    env.setdefault("GSS_NLTK_DIR", str(project_path / "nltk_data"))
    env["NLTK_DATA"] = env["GSS_NLTK_DIR"]

    log(f"[prepare] comando: {' '.join(cmd)}")
    proc = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr,
                          env=env, cwd=str(Path.cwd()))
    log(f"[prepare] finalizado com código {proc.returncode}")
    return proc.returncode


# ------------------------------------------------------------------
# 4. Inicia treino (gss_train.py) — igual ao gss_app.py
# ------------------------------------------------------------------
def pick_threads() -> int:
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        return 1 if ram_gb < 8.0 else 2
    except Exception:
        return 1


def start_train(project_path: Path, repo_path: Path,
                mode: str = "fast", run_name: str | None = None,
                batch_size: int = 8, epochs: int | None = None,
                threads: int = 2) -> int:
    threads = pick_threads()
    log("[train] chamando gss_train.py ...")
    # Verifica se dataset está preparado
    train_txt = project_path / "corpus" / "preprocessed_data" / "LJSpeech" / "train.txt"
    if not train_txt.is_file():
        log(f"[train] ERRO: dataset não preparado — {train_txt} não existe")
        return 1

    run = run_name or f"{mode}_{int(__import__('time').time())}"
    cmd = [
        sys.executable,
        str(BIN_DIR / "gss_train.py"),
        "--repo", str(repo_path),
        "--preprocess-config", str(project_path / "preprocess.yaml"),
        "--run-name", run,
        "--mode", mode,
        "--batch-size", str(batch_size),
        "--num-workers", "0",
        "--threads", str(threads),
        "--log-dir", str(project_path / "logs"),
    ]
    if epochs:
        cmd += ["--max-epochs", str(epochs)]
    if not __import__("torch").cuda.is_available():
        cmd += ["--precision", "32-true"]
    # Sempre sem val audio para ser rápido
    cmd += ["--no-val-audio"]

    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MALLOC_ARENA_MAX", "2")
    pp = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    for cand in (str(APP_DIR), str(repo_path / "grainspeech")):
        if cand not in pp:
            pp.append(cand)
    env["PYTHONPATH"] = os.pathsep.join(pp)
    env["GSS_APP_DIR"] = str(APP_DIR)
    env.setdefault("GSS_NLTK_DIR", str(project_path / "nltk_data"))
    env["NLTK_DATA"] = env["GSS_NLTK_DIR"]

    log(f"[train] comando: {' '.join(cmd)}")
    proc = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr,
                          env=env, cwd=str(Path.cwd()))
    log(f"[train] finalizado com código {proc.returncode}")
    return proc.returncode


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="GrainSpeech Studio — finetune rápido a partir de MP3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mp3", required=True,
                        help="arquivo .mp3 para upload")
    parser.add_argument("--project", default="finetune_project",
                        help="nome/pasta do projeto (default: finetune_project)")
    parser.add_argument("--repo", default=None,
                        help="pasta do clone do GrainSpeech")
    parser.add_argument("--run-name", default=None,
                        help="nome da execução")
    parser.add_argument("--mode", choices=["fast", "scratch"], default="fast",
                        help="modo de treino (default: fast)")
    parser.add_argument("--aligner", choices=["mfa", "uniform"], default="mfa",
                        help="alinhador (default: mfa)")
    parser.add_argument("--whisper-size", default="base.en",
                        help="tamanho do modelo Whisper (default: base.en)")
    parser.add_argument("--language", default="en",
                        help="idioma para Whisper (default: en)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    mp3_path = Path(args.mp3).expanduser().resolve()
    if not mp3_path.exists():
        log(f"[ERRO] arquivo MP3 não encontrado: {mp3_path}")
        return 1
    if mp3_path.suffix.lower() != ".mp3":
        log(f"[AVISO] arquivo não tem extensão .mp3: {mp3_path}")

    repo_path = Path(args.repo).resolve() if args.repo else REPO_DIR
    if not repo_path.exists():
        alt = APP_DIR.parent / "GrainSpeech"
        if alt.exists():
            repo_path = alt
        else:
            log(f"[ERRO] repositório GrainSpeech não encontrado em: {repo_path}")
            return 1

    project_arg = Path(args.project)
    if project_arg.is_absolute():
        project_path = project_arg
    else:
        project_path = APP_DIR.parent / args.project
    run_name = args.run_name or "mp3_finetune"
    project_path.mkdir(parents=True, exist_ok=True)
    (project_path / "logs").mkdir(exist_ok=True)
    (project_path / "outputs").mkdir(exist_ok=True)

    log("=" * 60)
    log(f"  GrainSpeech Studio — Finetune MP3")
    log(f"  MP3: {mp3_path}")
    log(f"  Projeto: {project_path}")
    log(f"  Modo: {args.mode}")
    log("=" * 60)

    # 1. Upload
    audio_dir = setup_audio_dir(mp3_path)

    # 2. Transcrição
    texto = transcribe_audio(mp3_path, audio_dir, language=args.language)

    # Se houver .lab manual criado, o gss_prepare já o usa automaticamente
    # Se não, continua com o texto retornado (Whisper/nltk/simulado)

    # 3. Prepara dataset
    rc_prepare = prepare_dataset(
        project_path=project_path,
        repo_path=repo_path,
        audio_dir=audio_dir,
        aligner=args.aligner,
        whisper_size=args.whisper_size,
        language=args.language,
        val_size=6,
    )
    if rc_prepare != 0:
        log("[prepare] falhou — continuando mesmo assim se possível ...")
        # Se falhou mas já temos arquivo de texto, pode ser aceitável

    # 4. Treino
    rc_train = start_train(
        project_path=project_path,
        repo_path=repo_path,
        mode=args.mode,
        run_name=run_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    log("=" * 60)
    if rc_train == 0:
        log("  Treino iniciado / finalizado com sucesso!")
    else:
        log(f"  Treino finalizado com código {rc_train} (verifique logs em {project_path / 'logs'})")
    log(f"  Projeto: {project_path}")
    log("=" * 60)
    return rc_train


if __name__ == "__main__":
    raise SystemExit(main())
