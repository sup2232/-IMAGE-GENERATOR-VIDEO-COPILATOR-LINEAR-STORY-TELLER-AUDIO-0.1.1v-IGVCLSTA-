"""
Etapa 1 do GrainSpeech Studio: transformar uma pasta de áudios em dataset
treinável.

    áudio (.wav/.mp3/.flac/...)
      -> wav 22.050 Hz mono + cauda de silêncio
      -> transcrição  (metadata.csv / .lab / .txt / Whisper)
      -> alinhamento forçado (MFA) ou fallback uniforme
      -> TextGrid/LJSpeech/*.TextGrid
      -> preprocess do GrainSpeech (mel, pitch, energy, duration, splits, stats)

Uso:
    python gss_prepare.py --root <projeto> --repo <GrainSpeech> \
        --audio-dir <pasta com áudios> [--aligner mfa|uniform]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP = Path(os.environ["GSS_APP_DIR"]) if os.environ.get("GSS_APP_DIR") else HERE.parent
for cand in (str(APP), str(APP.parent)):
    if cand not in sys.path:
        sys.path.insert(0, cand)

import numpy as np                                              # noqa: E402
import soundfile as sf                                          # noqa: E402
import yaml                                                     # noqa: E402
from gss import mfa_align, transcribe                           # noqa: E402
from gss.pipeline import Project                                # noqa: E402

SR = 22050
HOP = 256
NFFT = 1024
# O STFT do repo entrega floor((N + n_fft) / hop) frames, enquanto o
# pré-processamento conta frames como round(t * sr / hop). Sem folga no fim do
# áudio ele aborta com "requires N frames, but the audio features are shorter".
TAIL_SAMPLES = 2560          # ~116 ms -> ~10 frames de folga
PAUSE = {",", ";", ":", ".", "!", "?"}


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------
def collect_audio(audio_dir: Path) -> list[Path]:
    paths = transcribe.find_audio(audio_dir)
    log(f"[1/6] {len(paths)} arquivos de áudio em {audio_dir}")
    return paths


def load_transcripts(audio_dir: Path, paths: list[Path]) -> dict[str, str]:
    """metadata.csv (LJSpeech) > .lab > .txt por áudio. Vazio -> usa Whisper."""
    texts: dict[str, str] = {}
    metas = list(audio_dir.rglob("metadata.csv")) + list(audio_dir.rglob("metadata.tsv"))
    if metas:
        texts = transcribe.read_metadata(metas[0])
        log(f"[2/6] transcrições de {metas[0].name}: {len(texts)} linhas")
    if not texts:
        for p in paths:
            for ext in (".lab", ".txt"):
                c = p.with_suffix(ext)
                if c.is_file():
                    t = c.read_text(encoding="utf-8", errors="replace").strip()
                    if t:
                        texts[p.stem] = t
                        break
        if texts:
            log(f"[2/6] transcrições de arquivos .lab/.txt: {len(texts)}")
    return texts


def transcribe_missing(audio_dir: Path, paths: list[Path], texts: dict[str, str],
                       size: str, language: str, cache: Path) -> dict[str, str]:
    missing = [p for p in paths if p.stem not in texts]
    if not missing:
        return texts
    log(f"[2/6] {len(missing)} áudios sem transcrição -> Whisper '{size}'")
    model = transcribe.load_model(size, cache=cache)
    texts = dict(texts)
    for i, p in enumerate(missing, 1):
        segs, _ = model.transcribe(str(p), language=language,
                                   vad_filter=True, word_timestamps=False)
        t = " ".join(s.text for s in segs).strip()
        if t:
            texts[p.stem] = t
        if i % 10 == 0 or i == len(missing):
            log(f"      transcritos {i}/{len(missing)}")
    return texts


# --------------------------------------------------------------------------
def normalize_wavs(paths: list[Path], texts: dict[str, str], out_dir: Path,
                   max_seconds: float | None = None,
                   min_seconds: float = 0.4) -> tuple[dict[str, Path], dict[str, str]]:
    """22.05 kHz mono, pico normalizado e cauda de silêncio."""
    import librosa
    out_dir.mkdir(parents=True, exist_ok=True)
    wavs, kept, total = {}, {}, 0.0
    dropped = []
    for p in paths:
        uid = p.stem
        if uid not in texts or not texts[uid].strip():
            dropped.append((uid, "sem transcrição"))
            continue
        try:
            y, _ = librosa.load(str(p), sr=SR, mono=True)
        except Exception as exc:
            dropped.append((uid, f"falha ao ler: {exc}"))
            continue
        if len(y) / SR < min_seconds:
            dropped.append((uid, f"curto demais ({len(y)/SR:.2f}s)"))
            continue
        if max_seconds and len(y) / SR > max_seconds:
            y = y[: int(max_seconds * SR)]
        peak = float(np.max(np.abs(y))) or 1.0
        y = y / peak * 0.95
        y = np.concatenate([y, np.zeros(TAIL_SAMPLES, dtype=np.float32)])
        dst = out_dir / f"{uid}.wav"
        sf.write(str(dst), y, SR, subtype="PCM_16")
        wavs[uid] = dst
        kept[uid] = texts[uid].strip()
        total += len(y) / SR
    log(f"[3/6] {len(wavs)} wavs normalizados ({total/60:.2f} min) -> {out_dir}")
    for uid, why in dropped[:10]:
        log(f"      ignorado {uid}: {why}")
    if len(dropped) > 10:
        log(f"      ... e mais {len(dropped)-10}")
    if not wavs:
        raise SystemExit("nenhum par (áudio, transcrição) utilizável")
    return wavs, kept


def write_metadata(csv_path: Path, texts: dict[str, str]) -> None:
    transcribe.write_metadata(csv_path, texts)
    log(f"      metadata.csv: {len(texts)} linhas -> {csv_path}")


# --------------------------------------------------------------------------
def align_uniform(wavs: dict[str, Path], texts: dict[str, str], out_dir: Path,
                  supported: set[str]) -> int:
    """
    Fallback sem MFA: timestamps de PALAVRA do Whisper + fonemas distribuídos
    por peso de classe dentro de cada palavra. Não é alinhamento forçado de
    verdade; use apenas para smoke test.
    """
    from g2p_en import G2p
    g2p = G2p()
    out_dir.mkdir(parents=True, exist_ok=True)
    VOW = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY",
           "OW", "OY", "UH", "UW"}
    LIQ = {"L", "R", "W", "Y", "M", "N", "NG"}

    def weight(ph):
        b = ph.rstrip("012")
        if b in VOW:
            return 2.6
        if ph == "sp":
            return 1.6
        if b in LIQ:
            return 1.5
        return 1.0

    n_ok = 0
    for uid, wav in wavs.items():
        info = sf.info(str(wav))
        dur = info.frames / info.samplerate
        n_frames = max(1, int(dur * SR / HOP) - 12)
        phones = []
        for tok in g2p(texts[uid]):
            if tok in supported:
                phones.append(tok)
            elif tok in PAUSE and phones and phones[-1] != "sp":
                phones.append("sp")
        while phones and phones[-1] == "sp":
            phones.pop()
        phones = [p for p in phones if p != "sp"] or ["spn"]
        w = np.array([weight(p) for p in phones], dtype=np.float64)
        raw = w / w.sum() * n_frames
        counts = np.maximum(np.floor(raw).astype(int), 1)
        i = 0
        while counts.sum() != n_frames:
            counts[i % len(counts)] += 1 if counts.sum() < n_frames else -1
            i += 1
        FRAME = HOP / SR
        items = [("sil", 0.0, 4 * FRAME)]
        t = 4 * FRAME
        for ph, c in zip(phones, counts):
            items.append((ph, t, t + c * FRAME))
            t += c * FRAME
        items.append(("sil", t, dur))
        lines = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "",
                 "xmin = 0 ", f"xmax = {dur:.9f} ", "tiers? <exists> ",
                 "size = 1 ", "item []: ", "    item [1]: ",
                 '        class = "IntervalTier" ', '        name = "phones" ',
                 "        xmin = 0 ", f"        xmax = {dur:.9f} ",
                 f"        intervals: size = {len(items)} "]
        for i, (nm, a, b) in enumerate(items, 1):
            lines += [f"        intervals [{i}]: ", f"            xmin = {a:.9f} ",
                      f"            xmax = {b:.9f} ", f'            text = "{nm}" ']
        (out_dir / f"{uid}.TextGrid").write_text("\n".join(lines) + "\n",
                                                 encoding="utf-8")
        n_ok += 1
    return n_ok


# --------------------------------------------------------------------------
def run_preprocess(proj: Project, textgrid_dir: Path) -> int:
    """Chama o preprocess.py do repositório."""
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    if os.environ.get("NLTK_DATA"):
        env["NLTK_DATA"] = os.environ["NLTK_DATA"]
    pp = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    gs = str(proj.repo / "grainspeech")
    if gs not in pp:
        pp.append(gs)
    env["PYTHONPATH"] = os.pathsep.join(pp)
    cmd = [sys.executable, str(proj.repo / "grainspeech" / "preprocess.py"),
           "--preprocess-config", str(proj.config),
           "--textgrid-dir", str(textgrid_dir),
           "--device", "cpu"]
    log("[6/6] " + " ".join(cmd))
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    out = (proc.stdout + proc.stderr).replace("\r", "\n")
    for ln in out.splitlines():
        if ln.strip() and "it/s]" not in ln and "s/it]" not in ln:
            log("      " + ln.strip())
    if proc.returncode != 0:
        raise SystemExit(f"preprocess falhou (rc={proc.returncode})")
    return proc.returncode


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--aligner", choices=("mfa", "uniform"), default="mfa")
    ap.add_argument("--whisper-size", default="base.en")
    ap.add_argument("--language", default="en")
    ap.add_argument("--mfa-bin", default=None)
    ap.add_argument("--mfa-root", default=None)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--val-size", type=int, default=6)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--whisper-cache", default=None)
    args = ap.parse_args()

    proj = Project(root=Path(args.root), repo=Path(args.repo))
    audio_dir = Path(args.audio_dir)
    gs = proj.repo / "grainspeech"
    if str(gs) not in sys.path:
        sys.path.insert(0, str(gs))
    from text.symbols import symbols
    supported = {s[1:] for s in symbols if s.startswith("@")}
    # garante cmudict + tagger antes de instanciar o g2p_en
    sys.path.insert(0, str(APP / "bin"))
    from gss_infer import ensure_nltk_data
    nltk_dir = Path(os.environ.get("GSS_NLTK_DIR")
                    or os.environ.get("NLTK_DATA")
                    or (proj.root / "nltk_data"))
    ensure_nltk_data(nltk_dir)
    os.environ["NLTK_DATA"] = str(nltk_dir)
    log(f"      NLTK_DATA = {nltk_dir}")

    paths = collect_audio(audio_dir)
    texts = load_transcripts(audio_dir, paths)
    cache = Path(args.whisper_cache) if args.whisper_cache else (proj.workdir / "whisper_cache")
    texts = transcribe_missing(audio_dir, paths, texts, args.whisper_size,
                               args.language, cache)

    wavs, kept = normalize_wavs(paths, texts, proj.wavs, args.max_seconds)
    write_metadata(proj.metadata, kept)

    # ---- alinhamento -------------------------------------------------------
    aligner_used = args.aligner
    if aligner_used == "mfa":
        try:
            log("[4/6] alinhamento forçado com MFA ...")
            tg_out = mfa_align.align_dataset(
                wavs, kept, proj.workdir / "mfa",
                symbols_module_path=gs, mfa_bin=args.mfa_bin,
                mfa_root=Path(args.mfa_root) if args.mfa_root else None,
                jobs=args.jobs, log=log)
            final_tg = proj.textgrids
            if final_tg.exists():
                shutil.rmtree(final_tg)
            shutil.copytree(tg_out, final_tg)
            log(f"[5/6] TextGrids instalados em {final_tg}")
        except Exception as e:
            log(f"[4/6] MFA indisponível/falhou ({e}). Usando fallback alinhamento uniforme.")
            aligner_used = "uniform"

    if aligner_used != "mfa":
        log("[4/6] MFA desativado/fallback: alinhamento uniforme")
        final_tg = proj.textgrids
        n = align_uniform(wavs, kept, final_tg, supported)
        log(f"[5/6] {n} TextGrids aproximados -> {final_tg}")

    # ---- config + features -------------------------------------------------
    n_ut = len(list(proj.wavs.glob("*.wav")))
    val_size = max(1, min(args.val_size, n_ut // 8))
    if n_ut <= 2:
        val_size = 1
    cfg = proj.write_config(val_size=val_size)
    log(f"      preprocess.yaml -> {cfg} (val_size={val_size}, n={n_ut})")
    # install_textgrids() procura em <dir>/LJSpeech/*.TextGrid;
    # final_tg é .../corpus/TextGrid/LJSpeech -> passar .../corpus/TextGrid
    run_preprocess(proj, final_tg.parent)

    train_file = proj.preprocessed / "train.txt"
    val_file = proj.preprocessed / "val.txt"
    if train_file.exists():
        train_lines = [l for l in train_file.read_text().splitlines() if l.strip()]
    else:
        train_lines = []
    if val_file.exists():
        val_lines = [l for l in val_file.read_text().splitlines() if l.strip()]
    else:
        val_lines = []

    if not train_lines and val_lines:
        train_lines = val_lines[:]
        train_file.write_text("\n".join(train_lines) + "\n", encoding="utf-8")

    n_train = len(train_lines)
    n_val = len(val_lines)
    hours = 0.0
    for w in proj.wavs.glob("*.wav"):
        hours += sf.info(str(w)).frames / SR / 3600
    proj.save_state(
        n_utterances=n_ut, n_train=n_train, n_val=n_val,
        minutes=round(hours * 60, 2), aligner=args.aligner,
        whisper_size=args.whisper_size, val_size=val_size,
        stats=str(proj.stats), config=str(proj.config),
        prepared=True,
    )
    log("")
    log("=" * 66)
    log(f"  dataset pronto: {n_ut} utterances | {hours*60:.2f} min | "
        f"alinhador={args.aligner}")
    log(f"  treino {n_train} / validação {n_val}")
    log(f"  stats.json: {proj.stats}")
    log("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
