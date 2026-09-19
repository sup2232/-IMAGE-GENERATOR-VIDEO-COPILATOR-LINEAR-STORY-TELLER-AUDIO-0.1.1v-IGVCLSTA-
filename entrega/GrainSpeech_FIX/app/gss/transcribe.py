"""
Transcrição automática dos áudios com faster-whisper.

Sai um `metadata.csv` no formato LJSpeech que o GrainSpeech usa:

    basename|texto original|texto normalizado

O faster-whisper (CTranslate2) é usado em vez do `openai-whisper` porque:
  * instala por pip puro (sem compilar nada) no Windows e no Linux;
  * é 3-5x mais rápido em CPU com int8;
  * devolve timestamps por palavra (`word_timestamps=True`), que servem
    de fallback quando o MFA não está disponível.
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path

AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".wma", ".aac"}
_PUNCT = re.compile(r"[^\w\s'.,!?-]", re.UNICODE)


def normalize_ljspeech(text: str) -> str:
    """Texto normalizado no estilo LJSpeech (minúsculas, sem pontuação)."""
    text = _PUNCT.sub(" ", text.lower())
    return " ".join(text.split())


def find_audio(folder: Path) -> list[Path]:
    folder = Path(folder)
    out = [p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_EXT]
    return sorted(out)


def load_model(size: str = "base.en", cache: Path | None = None,
               device: str = "auto", compute_type: str = "int8"):
    """Carrega o modelo Whisper (download automático no primeiro uso)."""
    from faster_whisper import WhisperModel
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    cache = Path(cache) if cache else Path(os.getcwd()) / ".cache" / "whisper"
    cache.mkdir(parents=True, exist_ok=True)
    if device == "cuda":
        compute_type = "float16"
    return WhisperModel(size, device=device, compute_type=compute_type,
                        download_root=str(cache))


def transcribe_folder(folder: Path, model=None, size: str = "base.en",
                      language: str = "en", cache: Path | None = None,
                      log=print) -> tuple[dict[str, Path], dict[str, str], dict[str, list]]:
    folder = Path(folder)
    paths = find_audio(folder)
    if not paths:
        raise FileNotFoundError(f"nenhum áudio encontrado em {folder}")
    log(f"{len(paths)} arquivos de áudio em {folder}")

    own_model = False
    if model is None:
        log(f"carregando Whisper '{size}' ...")
        model = load_model(size, cache=cache)
        own_model = True

    wavs, texts, stamps = {}, {}, {}
    try:
        for i, p in enumerate(paths, 1):
            uid = p.stem
            segments, _info = model.transcribe(str(p), language=language,
                                              word_timestamps=True,
                                              vad_filter=True)
            buf, words = [], []
            for seg in segments:
                buf.append(seg.text)
                for w in (seg.words or []):
                    words.append((float(w.start), float(w.end), w.word.strip()))
            text = " ".join(" ".join(buf).split()).strip()
            if not text:
                log(f"  ⚠ {uid}: transcrição vazia, ignorando")
                continue
            wavs[uid] = p
            texts[uid] = text
            stamps[uid] = words
            if i % 10 == 0 or i == len(paths):
                log(f"  transcritos {i}/{len(paths)}")
    finally:
        if own_model:
            del model
            import gc
            gc.collect()
            try:
                import ctypes
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass

    return wavs, texts, stamps


def write_metadata(out_csv: Path, texts: dict[str, str]) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="|", quoting=csv.QUOTE_MINIMAL,
                       lineterminator="\n")
        for uid in sorted(texts):
            w.writerow([uid, texts[uid], normalize_ljspeech(texts[uid])])


def read_metadata(meta_csv: Path) -> dict[str, str]:
    """Lê um metadata.csv LJSpeech (2 ou 3 colunas, separador '|')."""
    out = {}
    with Path(meta_csv).open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 2 or not parts[0]:
                continue
            text = parts[2] if len(parts) >= 3 and parts[2].strip() else parts[1]
            out[parts[0]] = text
    return out
