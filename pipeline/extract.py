"""Étape 1 — Extraction des frames et de l'audio avec FFmpeg."""
from __future__ import annotations

from pathlib import Path

import config
from .utils import VideoInfo, probe, reset_dir, run


def extract_first_frame(video_path: str | Path, out_path: str | Path) -> str:
    """Extrait uniquement la première frame (pour l'aperçu / le dessin de la zone)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", "select=eq(n\\,0)", "-vframes", "1",
        str(out_path),
    ])
    return str(out_path)


def extract_last_frame(video_path: str | Path, out_path: str | Path) -> str:
    """Extrait approximativement la dernière frame (pour le filigrane mobile)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # -sseof -3 : se place ~3s avant la fin, -update 1 garde la dernière image décodée.
    run([
        "ffmpeg", "-y", "-sseof", "-3", "-i", str(video_path),
        "-update", "1", "-q:v", "1", str(out_path),
    ], check=False)
    if not out_path.exists():
        # Repli : première frame si l'extraction de fin échoue.
        return extract_first_frame(video_path, out_path)
    return str(out_path)


def extract_frames(video_path: str | Path) -> VideoInfo:
    """
    Extrait toutes les frames de la vidéo en PNG dans config.FRAMES_DIR,
    et l'audio original dans config.AUDIO_PATH (si présent).

    Renvoie les métadonnées de la vidéo.
    """
    info = probe(video_path)

    reset_dir(config.FRAMES_DIR)
    # Frames nommées frame_000001.png, frame_000002.png, ...
    run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-start_number", "0",
        str(Path(config.FRAMES_DIR) / "frame_%06d.png"),
    ])

    # Audio extrait à part pour le remux final (copie sans réencodage).
    if info.has_audio:
        config.AUDIO_PATH.parent.mkdir(parents=True, exist_ok=True)
        if config.AUDIO_PATH.exists():
            config.AUDIO_PATH.unlink()
        run([
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-acodec", "copy", str(config.AUDIO_PATH),
        ], check=False)  # certains conteneurs refusent la copie directe -> non bloquant

    return info
