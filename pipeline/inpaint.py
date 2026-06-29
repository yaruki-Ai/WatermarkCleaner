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

    # On réduit la résolution de traitement pour tenir dans la RAM de Colab.
    ratio = config.resize_ratio_for(info.width, info.height)
    proc_w = int(round(info.width * ratio))
    proc_h = int(round(info.height * ratio))
    subvideo_len = config.subvideo_length_for(proc_w, proc_h)
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
    if ratio < 1.0:
        cmd += ["--resize_ratio", str(ratio)]
    if config.USE_FP16:
        cmd.append("--fp16")

    if progress:
        res_msg = f"{proc_w}x{proc_h}" if ratio < 1.0 else "résolution d'origine"
        progress(0.3, f"ProPainter en cours ({res_msg}, chunks de {subvideo_len} frames)… "
                      "Étape longue, sois patient.")

    # cwd = dossier ProPainter pour qu'il trouve ses poids/relatifs.
    # On capture TOUTE la sortie (stdout + stderr) pour un diagnostic fiable.
    proc = subprocess.run(
        cmd, cwd=str(pp_dir),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    )

    if proc.returncode != 0:
        log = (proc.stdout or "") + "\n" + (proc.stderr or "")
        log = log.strip() or "(aucune sortie capturée — voir la sortie de la cellule Colab)"
        print("===== SORTIE PROPAINTER (échec) =====")
        print(log)
        print("=====================================")
        low = log.lower()
        # code -9 = tué par le système (SIGKILL), quasi toujours un manque de RAM.
        if proc.returncode == -9 or "killed" in low:
            raise RuntimeError(
                "❌ Manque de mémoire : Colab a tué le traitement (code -9). "
                f"Résolution de traitement utilisée : {proc_w}x{proc_h}. "
                "Raccourcis la vidéo, ou réduis encore MAX_PROCESS_SIDE dans config.py."
            )
        if "out of memory" in low or "cuda oom" in low:
            raise RuntimeError(
                "❌ Dépassement mémoire GPU (out-of-memory). "
                "Réduis la résolution ou raccourcis la vidéo, puis réessaie."
            )
        # On renvoie les dernières lignes dans l'UI (les plus parlantes).
        tail = "\n".join(log.splitlines()[-15:])
        raise RuntimeError("❌ ProPainter a échoué (code "
                           f"{proc.returncode}) :\n{tail}")

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
