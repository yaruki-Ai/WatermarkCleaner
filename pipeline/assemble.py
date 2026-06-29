"""Étape 4 — Réassemblage : frames nettoyées -> vidéo + remux de l'audio original."""
from __future__ import annotations

from pathlib import Path

import config
from .utils import VideoInfo, list_frames, run


def finalize(frames_dir: str | Path, info: VideoInfo, out_name: str) -> str:
    """
    Reconstruit la vidéo depuis le dossier de frames nettoyées, recolle l'audio
    original, et écrit le résultat final sur Google Drive (config.RESULTS_DIR).

    Renvoie le chemin de la vidéo finale.
    """
    frames_dir = Path(frames_dir)
    frames = list_frames(frames_dir)
    if not frames:
        raise RuntimeError(f"Aucune frame nettoyée trouvée dans {frames_dir}.")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULTS_DIR / out_name

    # Vidéo silencieuse depuis les frames (frame_000000.png, frame_000001.png, …).
    silent = Path(config.WORK_DIR) / "inpainted_silent.mp4"
    if silent.exists():
        silent.unlink()
    run([
        "ffmpeg", "-y",
        "-framerate", f"{info.fps}",
        "-start_number", "0",
        "-i", str(frames_dir / "frame_%06d.png"),
        # H.264/yuv420p exige des dimensions paires -> on arrondit au pair inférieur.
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "16",  # haute qualité
        str(silent),
    ])

    has_audio = info.has_audio and Path(config.AUDIO_PATH).exists()
    if has_audio:
        run([
            "ffmpeg", "-y",
            "-i", str(silent),
            "-i", str(config.AUDIO_PATH),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            str(out_path),
        ])
    else:
        import shutil
        shutil.copy2(str(silent), str(out_path))

    return str(out_path)
