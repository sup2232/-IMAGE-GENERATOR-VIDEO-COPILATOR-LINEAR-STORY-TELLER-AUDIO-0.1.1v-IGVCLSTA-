"""
Treino do GrainSpeech com dados próprios.

Dois modos:

  --mode fast     fine-tune a partir do checkpoint oficial (264.881 parâmetros
                  do `phoneme2mel` são copiados; o HiFi-GAN já vem congelado).
                  Convergência em dezenas de épocas mesmo com minutos de áudio.

  --mode scratch  treino do zero, como no paper (LJSpeech, 5000 épocas).

Também corrige dois problemas do script de treino original do repositório:

  1. `get_lr_scheduler(optimizer, 50, self.hparams.max_epochs)` passa
     *max_epochs* onde a função espera *total_steps*. Como o Lightning avança
     o scheduler por batch, o decaimento cossenoidal termina em
     `max_epochs` BATCHES, não épocas. Com 100 épocas e ~3 batches/época o
     cosine chega a ~0 no batch 50, ou seja, na época 17. Aqui o total de
     passos é calculado de verdade.

  2. Não existe opção de inicializar a partir do checkpoint publicado — o
     `--checkpoint` do repo só retoma um estado Lightning completo. Aqui o
     state_dict `phoneme2mel.*` é carregado com `strict=False`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml


def _add_repo_to_path(grainspeech_dir: Path) -> None:
    for p in (str(grainspeech_dir), str(grainspeech_dir.parent)):
        if p not in sys.path:
            sys.path.insert(0, p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True,
                    help="pasta do clone do GrainSpeech (contém grainspeech/)")
    ap.add_argument("--preprocess-config", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--mode", choices=("fast", "scratch"), default="fast")
    ap.add_argument("--official-ckpt", default=None,
                    help="checkpoints/grainspeech_l1_ssim_gvar.ckpt (default: <repo>/checkpoints/...)")
    ap.add_argument("--hifigan-checkpoint", default=None,
                    help="default: <repo>/hifigan/LJ_V2/generator_v2")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--precision", default="32-true",
                    help="'16-mixed' só em GPU; em CPU use '32-true'")
    ap.add_argument("--accelerator", default="auto")
    ap.add_argument("--val-size", type=int, default=6,
                    help="apenas informativo; quem decide é o preprocess.yaml")
    ap.add_argument("--log-dir", default="lightning_logs")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--limit-train-batches", type=float, default=1.0,
                    help="fração do treino por época (smoke test rápido)")
    ap.add_argument("--check-val-every-n-epoch", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--no-val-audio", action="store_true",
                    help="não rodar o vocoder na validação (economiza muita RAM)")
    ap.add_argument("--threads", type=int, default=0,
                    help="torch.set_num_threads (0 = não mexer)")
    ap.add_argument("--resume", default=None,
                    help="retomar um checkpoint Lightning completo (.ckpt) do próprio treino")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    gs = repo / "grainspeech"
    if not gs.is_dir():
        raise SystemExit(f"não achei {gs}")
    _add_repo_to_path(gs)

    from lightning import Trainer, seed_everything
    from lightning.pytorch.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
    from lightning.pytorch.loggers import TensorBoardLogger

    import model_l1_ssim_gvar as M
    from datamodule import LJSpeechDataModule

    if args.threads:
        torch.set_num_threads(args.threads)
    seed_everything(args.seed, workers=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    accelerator = args.accelerator
    if accelerator == "auto":
        accelerator = "gpu" if device == "cuda" else "cpu"
    if accelerator == "gpu" and device != "cuda":
        accelerator = "cpu"

    official = Path(args.official_ckpt or (repo / "checkpoints" /
                                           "grainspeech_l1_ssim_gvar.ckpt"))
    vocoder = str(args.hifigan_checkpoint or (repo / "hifigan" / "LJ_V2" / "generator_v2"))

    with open(args.preprocess_config, encoding="utf-8") as fh:
        preprocess_config = yaml.safe_load(fh)

    # ---------------------------------------------------------------- defaults por modo
    if args.mode == "fast":
        max_epochs = args.max_epochs if args.max_epochs is not None else 120
        lr = args.lr if args.lr is not None else 2e-4
    else:
        max_epochs = args.max_epochs if args.max_epochs is not None else 5000
        lr = args.lr if args.lr is not None else 1e-3

    print("=" * 72)
    print(f"  GrainSpeech | modo={args.mode} | device={device} | precision={args.precision}")
    print(f"  config      : {args.preprocess_config}")
    print(f"  epochs={max_epochs}  batch={args.batch_size}  lr={lr}  wd={args.weight_decay}")
    print("=" * 72)

    datamodule = LJSpeechDataModule(preprocess_config=preprocess_config,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers)
    datamodule.setup()
    n_train = len(datamodule.train_dataset)
    n_val = len(datamodule.test_dataset)
    steps_per_epoch = max(1, n_train // args.batch_size)
    total_steps = steps_per_epoch * max_epochs
    print(f"  dados: {n_train} treino / {n_val} validação  "
          f"-> {steps_per_epoch} steps/época, {total_steps} steps no total")

    # ---------------------------------------------------------------- CORREÇÃO 1: LR schedule
    warmup = min(50, max(1, total_steps // 20))

    def _patched_configure_optimizers(self):
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import LambdaLR
        import math
        optimizer = AdamW([p for p in self.parameters() if p.requires_grad],
                          lr=self.hparams.lr,
                          weight_decay=self.hparams.weight_decay)
        total = max(1, int(os.environ.get("GSS_TOTAL_STEPS", total_steps)))
        wu = int(os.environ.get("GSS_WARMUP_STEPS", warmup))

        def lam(step):
            if step < wu:
                return float(step) / float(max(1, wu))
            prog = float(step - wu) / float(max(1, total - wu))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog))))

        scheduler = LambdaLR(optimizer, lr_lambda=lam)
        self.scheduler = scheduler
        return [optimizer], [scheduler]

    os.environ["GSS_TOTAL_STEPS"] = str(total_steps)
    os.environ["GSS_WARMUP_STEPS"] = str(warmup)
    M.GrainSpeech.configure_optimizers = _patched_configure_optimizers

    # Em máquinas com pouca RAM o vocoder na validação é o pico de memória.
    # O loss de validação continua sendo calculado normalmente.
    if args.no_val_audio:
        _orig_vs = M.GrainSpeech.validation_step

        def _vs_no_audio(self, batch, batch_idx):
            prev, self.training = self.training, True
            try:
                return _orig_vs(self, batch, batch_idx)
            finally:
                self.training = prev

        def _plain_vs(self, batch, batch_idx):
            x, y = batch
            y_hat = self.phoneme2mel(x, train=True)
            (mel_loss, l1, ssim, gvar, pl, el, dl) = self.loss(y_hat, y, x)
            val_loss = (self.mel_weight * mel_loss + self.pitch_weight * pl
                        + self.energy_weight * el + self.duration_weight * dl)
            self.log_dict({"val_loss": val_loss, "val_l1": l1, "val_mel": mel_loss},
                          on_step=False, on_epoch=True, prog_bar=False,
                          batch_size=y["mel"].shape[0])

        M.GrainSpeech.validation_step = _plain_vs

    model = M.GrainSpeech(
        preprocess_config=preprocess_config,
        lr=lr,
        weight_decay=args.weight_decay,
        max_epochs=max_epochs,
        wav_path=str(Path(args.log_dir) / args.run_name / "val_wavs"),
        hifigan_checkpoint=vocoder,
        infer_device=device,
        verbose=False,
        constant_lr=False,
        mel_weight=5.0, l1_weight=1.0, ssim_weight=1.0, gvar_weight=0.5,
    )

    n_acoustic = sum(p.numel() for n, p in model.named_parameters()
                     if n.startswith("phoneme2mel."))
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  parâmetros acústicos: {n_acoustic:,} | treináveis: {n_trainable:,}")

    # ---------------------------------------------------------------- CORREÇÃO 2: fast start
    init_from = None
    if args.mode == "fast":
        if not official.is_file():
            raise SystemExit(f"checkpoint oficial não encontrado: {official}")
        ck = torch.load(official, map_location="cpu", weights_only=False)
        sd = ck["state_dict"] if "state_dict" in ck else ck
        wanted = {k: v for k, v in sd.items() if k.startswith("phoneme2mel.")}
        del ck
        import gc
        gc.collect()
        missing, unexpected = model.load_state_dict(wanted, strict=False)
        del wanted
        gc.collect()
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        got = len(sd) if "sd" in locals() else 0
        print(f"  ✓ fine-tune: tensores carregados de {official.name}")
        if unexpected:
            print(f"    inesperados: {len(unexpected)}")
        real_missing = [k for k in missing if k.startswith("phoneme2mel.")]
        if real_missing:
            print(f"    ⚠ faltando no phoneme2mel: {real_missing[:6]}")
        init_from = str(official)

    try:
        logger = TensorBoardLogger(save_dir=str(Path(args.log_dir).parent),
                                   name=Path(args.log_dir).name,
                                   version=args.run_name)
    except ModuleNotFoundError:      # sem tensorboard instalado -> CSV logger
        from lightning.pytorch.loggers import CSVLogger
        logger = CSVLogger(save_dir=str(Path(args.log_dir).parent),
                           name=Path(args.log_dir).name, version=args.run_name)
        print("  (tensorboard ausente: usando CSVLogger)")
    ckpt_cb = ModelCheckpoint(dirpath=os.path.join(logger.log_dir, "checkpoints"),
                              filename="epoch{epoch:04d}-loss{loss:.4f}",
                              auto_insert_metric_name=False,
                              monitor="loss", mode="min",
                              every_n_epochs=max(1, max_epochs // 10),
                              save_top_k=2, save_last=True)
    class JsonlMetrics(Callback):
        """Métricas por época em JSONL — a interface lê daqui (barato e sem TB)."""

        def __init__(self, path):
            self.path = Path(path)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")
            self.t0 = time.time()

        def _w(self, pl, extra=None):
            rec = {"epoch": int(pl.current_epoch), "step": int(pl.global_step),
                   "t": round(time.time() - self.t0, 1)}
            if extra:
                rec.update(extra)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        def on_train_epoch_end(self, trainer, pl):
            out = {}
            for k, v in pl.trainer.callback_metrics.items():
                try:
                    out[str(k)] = round(float(v), 6)
                except Exception:
                    pass
            out["lr"] = round(float(pl.trainer.optimizers[0].param_groups[0]["lr"]), 8)
            self._w(pl, out)

        def on_validation_epoch_end(self, trainer, pl):
            cm = pl.trainer.callback_metrics
            if "val_loss" in cm:
                self._w(pl, {"val_loss": round(float(cm["val_loss"]), 6),
                             "val_l1": round(float(cm.get("val_l1", float("nan"))), 6)})

    metrics_path = Path(logger.log_dir) / "metrics.jsonl"
    trainer = Trainer(
        accelerator=accelerator, devices=1, precision=args.precision,
        max_epochs=max_epochs, logger=logger,
        callbacks=[ckpt_cb, LearningRateMonitor(logging_interval="epoch"),
                   JsonlMetrics(metrics_path)],
        check_val_every_n_epoch=(args.check_val_every_n_epoch
                                 if args.check_val_every_n_epoch
                                 else max(1, max_epochs // 10)),
        num_sanity_val_steps=0,
        limit_train_batches=args.limit_train_batches,
        max_steps=args.max_steps,
        enable_progress_bar=True,
        log_every_n_steps=max(1, steps_per_epoch),
    )

    t0 = dt.datetime.now()
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)
    elapsed = dt.datetime.now() - t0

    last = ckpt_cb.last_model_path or ""
    best = ckpt_cb.best_model_path or ""
    stats_path = Path(preprocess_config["path"]["preprocessed_path"]) / "stats.json"
    summary = {
        "run_name": args.run_name, "mode": args.mode,
        "device": device, "precision": str(args.precision),
        "max_epochs": max_epochs, "batch_size": args.batch_size, "lr": lr,
        "n_train": n_train, "n_val": n_val,
        "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
        "acoustic_params": n_acoustic, "trainable_params": n_trainable,
        "init_from": init_from,
        "seconds_per_epoch": round(elapsed.total_seconds() / max(1, max_epochs), 3),
        "train_time": str(elapsed),
        "last_ckpt": last, "best_ckpt": best,
        "stats_json": str(stats_path),
        "log_dir": logger.log_dir,
        "metrics_jsonl": str(metrics_path),
    }
    out = Path(logger.log_dir) / "train_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + "=" * 72)
    print(f"  tempo total : {elapsed}")
    print(f"  s/época     : {summary['seconds_per_epoch']}")
    print(f"  best ckpt   : {best}")
    print(f"  last ckpt   : {last}")
    print(f"  stats p/ inf: {stats_path}")
    print(f"  resumo      : {out}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
