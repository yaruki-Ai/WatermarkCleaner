"""Étape 4 — Réassemblage final : remux de l'audio original sur la vidéo nettoyée."""
from __future__ import annotations

import shutil
from pathlib import Path

import config
from .utils import VideoInfo, run


def finalize(inpainted_video: str | Path, info: VideoInfo, out_name: str) -> str:
    """
    Recolle l'audio original sur la vidéo inpaintée et écrit le résultat final
    sur Google Drive (config.RESULTS_DIR).

    Renvoie le chemin de la vidéo finale.
    """
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULTS_DIR / out_name

    has_audio = info.has_audio and Path(config.AUDIO_PATH).exists()

    if has_audio:
        # Mux : flux vidéo de la sortie ProPainter + flux audio original.
        # -shortest pour aligner sur la plus courte des deux pistes.
        run([
            "ffmpeg", "-y",
            "-i", str(inpainted_video),
            "-i", str(config.AUDIO_PATH),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            str(out_path),
        ])
    else:
        # Pas d'audio : simple copie de la vidéo nettoyée vers Drive.
        shutil.copy2(str(inpainted_video), str(out_path))

    return str(out_path)
