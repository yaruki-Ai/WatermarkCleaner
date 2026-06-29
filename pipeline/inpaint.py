"""
Étape 3 — Inpainting vidéo avec ProPainter, par segments.

On appelle le script officiel `inference_propainter.py` en sous-processus.
Pour garder la pleine qualité sans saturer la RAM de Colab, on traite la vidéo
par tranches de `SEGMENT_SIZE` frames (avec recouvrement), chaque tranche dans
un appel ProPainter séparé. Les frames nettoyées sont ensuite réassemblées.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

import config
from .utils import VideoInfo, list_frames, reset_dir, run

# Callback de progression : reçoit (fraction_0_a_1, message).
ProgressFn = Callable[[float, str], None]


def feasibility_warning(info: VideoInfo) -> Optional[str]:
    """Renvoie un avertissement informatif (le traitement par segments gère la RAM)."""
    if info.n_frames > 2000:
        return (
            f"⚠️ Vidéo longue (~{info.n_frames} frames). "
            "Le traitement par segments fonctionnera mais sera long."
        )
    return None


def _check_propainter() -> Path:
    pp_dir = Path(config.PROPAINTER_DIR)
    if not (pp_dir / "inference_propainter.py").exists():
        raise RuntimeError(
            f"ProPainter introuvable dans {pp_dir}. "
            "Lance d'abord la cellule de clonage de ProPainter dans le notebook."
        )
    return pp_dir


def _run_pp_segment(pp_dir, in_frames, in_masks, out_root, ratio, subvideo_len, fps):
    """Lance ProPainter sur une tranche et renvoie la liste triée des frames nettoyées."""
    out_root = Path(out_root)
    reset_dir(out_root)

    cmd = [
        sys.executable, str(pp_dir / "inference_propainter.py"),
        "--video", str(Path(in_frames).resolve()),
        "--mask", str(Path(in_masks).resolve()),
        "--output", str(out_root.resolve()),
        "--subvideo_length", str(subvideo_len),
        "--save_fps", str(int(round(fps))),
        "--save_frames",
    ]
    if ratio < 1.0:
        cmd += ["--resize_ratio", str(ratio)]
    if config.USE_FP16:
        cmd.append("--fp16")

    proc = subprocess.run(cmd, cwd=str(pp_dir),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if proc.returncode != 0:
        log = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        log = log or "(aucune sortie capturée)"
        print("===== SORTIE PROPAINTER (échec) =====\n" + log + "\n=====================================")
        low = log.lower()
        if proc.returncode == -9 or "killed" in low:
            raise RuntimeError(
                "❌ Manque de mémoire (code -9) même par segments. "
                "Réduis SEGMENT_SIZE ou MAX_PROCESS_SIDE dans config.py."
            )
        if "out of memory" in low or "cuda oom" in low:
            raise RuntimeError("❌ Mémoire GPU insuffisante. Réduis MAX_PROCESS_SIDE dans config.py.")
        raise RuntimeError("❌ ProPainter a échoué (code "
                           f"{proc.returncode}) :\n" + "\n".join(log.splitlines()[-15:]))

    # ProPainter écrit la sortie dans out_root/<nom_du_dossier_entree>/...
    seg_out = out_root / Path(in_frames).name
    pngs = sorted(seg_out.rglob("*.png")) if seg_out.exists() else []
    if not pngs:
        # Repli : extrait les frames depuis inpaint_out.mp4 si pas de PNG.
        mp4s = list(out_root.rglob("inpaint_out.mp4"))
        if not mp4s:
            raise RuntimeError("ProPainter n'a produit aucune frame de sortie.")
        extract_dir = reset_dir(out_root / "extracted")
        run(["ffmpeg", "-y", "-i", str(mp4s[0]),
             str(extract_dir / "f_%06d.png")])
        pngs = sorted(extract_dir.glob("*.png"))
    return pngs


def run_propainter(info: VideoInfo, progress: Optional[ProgressFn] = None) -> str:
    """
    Inpainting par segments. Renvoie le chemin du dossier des frames nettoyées
    (prêtes à être réassemblées en vidéo).
    """
    pp_dir = _check_propainter()

    frames = list_frames(config.FRAMES_DIR)
    masks = list_frames(config.MASKS_DIR)
    n = len(frames)
    if n == 0:
        raise RuntimeError("Aucune frame à traiter.")
    if len(masks) != n:
        raise RuntimeError(f"Incohérence frames/masques ({n} vs {len(masks)}).")

    ratio = config.resize_ratio_for(info.width, info.height)
    proc_w, proc_h = int(round(info.width * ratio)), int(round(info.height * ratio))
    subvideo_len = config.subvideo_length_for(proc_w, proc_h)

    seg = max(8, config.SEGMENT_SIZE)
    overlap = max(0, min(config.SEGMENT_OVERLAP, seg - 1))

    final_dir = reset_dir(Path(config.WORK_DIR) / "final_frames")
    seg_in_frames = Path(config.WORK_DIR) / "seg_in"
    seg_in_masks = Path(config.WORK_DIR) / "seg_mask"
    seg_out_root = Path(config.WORK_DIR) / "seg_out"

    out_idx = 0
    start = 0
    res_msg = f"{proc_w}x{proc_h}" if ratio < 1.0 else "résolution native"

    while start < n:
        end = min(n, start + seg)
        if progress:
            frac = start / max(1, n)
            progress(frac, f"Inpainting {res_msg} — frames {start + 1}-{end}/{n}…")

        # Prépare les dossiers d'entrée de la tranche (noms uniques -> sortie unique).
        fin = reset_dir(seg_in_frames)
        min_ = reset_dir(seg_in_masks)
        for j in range(start, end):
            shutil.copy2(frames[j], fin / frames[j].name)
            shutil.copy2(masks[j], min_ / masks[j].name)

        out_pngs = _run_pp_segment(pp_dir, fin, min_, seg_out_root,
                                   ratio, subvideo_len, info.fps)

        # Pour les tranches après la 1re, on saute les frames de recouvrement
        # (déjà produites par la tranche précédente).
        skip = 0 if start == 0 else overlap
        for k in range(skip, len(out_pngs)):
            shutil.copy2(out_pngs[k], final_dir / f"frame_{out_idx:06d}.png")
            out_idx += 1

        if end >= n:
            break
        start = end - overlap

    if progress:
        progress(0.99, f"Inpainting terminé ({out_idx} frames).")
    return str(final_dir)
