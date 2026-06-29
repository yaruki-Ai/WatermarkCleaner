"""
Étape 3 — Inpainting vidéo avec ProPainter.

On appelle le script officiel `inference_propainter.py` en sous-processus.
ProPainter gère lui-même le découpage temporel via `--subvideo_length`, qu'on
règle selon la résolution pour tenir dans la VRAM du T4 (~15 Go).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

import config
from .utils import VideoInfo

# Callback de progression : reçoit (fraction_0_a_1, message).
ProgressFn = Callable[[float, str], None]


def feasibility_warning(info: VideoInfo) -> Optional[str]:
    """Renvoie un avertissement si la vidéo risque de dépasser la VRAM, sinon None."""
    pixels = info.width * info.height
    if pixels > config.MAX_SAFE_PIXELS:
        return (
            f"⚠️ Résolution très élevée ({info.width}x{info.height}). "
            "Le GPU T4 risque un dépassement mémoire (out-of-memory). "
            "Réduis la résolution de la vidéo (ex. 1080p max) ou raccourcis-la."
        )
    if info.n_frames > 1500:
        return (
            f"⚠️ Vidéo longue (~{info.n_frames} frames). "
            "Le traitement sera lent ; pense à découper la vidéo si ça échoue."
        )
    return None


def run_propainter(info: VideoInfo, progress: Optional[ProgressFn] = None) -> str:
    """
    Lance ProPainter sur les frames + masques.

    Renvoie le chemin de la vidéo inpaintée (sans audio) produite par ProPainter.
    Lève une RuntimeError avec un message clair en cas d'OOM ou d'échec.
    """
    pp_dir = Path(config.PROPAINTER_DIR)
    script = pp_dir / "inference_propainter.py"
    if not script.exists():
        raise RuntimeError(
            f"ProPainter introuvable dans {pp_dir}. "
            "Lance d'abord la cellule de clonage de ProPainter dans le notebook."
        )

    subvideo_len = config.subvideo_length_for(info.width, info.height)
    out_root = Path(config.PROPAINTER_OUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(script),
        "--video", str(Path(config.FRAMES_DIR).resolve()),
        "--mask", str(Path(config.MASKS_DIR).resolve()),
        "--output", str(out_root.resolve()),
        "--subvideo_length", str(subvideo_len),
        "--save_fps", str(int(round(info.fps))),
        "--save_frames",
    ]
    if config.USE_FP16:
        cmd.append("--fp16")

    if progress:
        progress(0.05, f"ProPainter démarre (chunks de {subvideo_len} frames, fp16={config.USE_FP16})…")

    # cwd = dossier ProPainter pour qu'il trouve ses poids/relatifs.
    proc = subprocess.Popen(
        cmd, cwd=str(pp_dir),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        tail.append(line)
        tail[:] = tail[-40:]  # garde les 40 dernières lignes pour le diagnostic
        if progress:
            # Progression grossière : on relaie la dernière ligne utile.
            progress(0.5, line[:120])

    proc.wait()

    if proc.returncode != 0:
        log = "\n".join(tail)
        if "out of memory" in log.lower() or "cuda oom" in log.lower():
            raise RuntimeError(
                "❌ Dépassement mémoire GPU (out-of-memory). "
                "La vidéo est trop grosse pour le T4. Réduis la résolution "
                "(1080p ou 720p) ou raccourcis la vidéo, puis réessaie.\n\n" + log
            )
        raise RuntimeError("❌ ProPainter a échoué :\n" + log)

    # ProPainter écrit dans {output}/{nom_du_dossier_video}/inpaint_out.mp4.
    # Le dossier d'entrée s'appelle "frames" -> sortie dans pp_out/frames/.
    result = out_root / Path(config.FRAMES_DIR).name / "inpaint_out.mp4"
    if not result.exists():
        # Repli : cherche n'importe quel inpaint_out.mp4 produit.
        candidates = list(out_root.rglob("inpaint_out.mp4"))
        if not candidates:
            raise RuntimeError(
                "ProPainter s'est terminé mais aucune vidéo de sortie n'a été trouvée."
            )
        result = candidates[0]

    if progress:
        progress(0.9, "Inpainting terminé.")
    return str(result)
