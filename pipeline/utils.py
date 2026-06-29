"""Helpers communs : ffprobe, exécution de commandes, nettoyage de dossiers."""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class VideoInfo:
    """Métadonnées d'une vidéo, lues via ffprobe."""
    width: int
    height: int
    fps: float
    duration: float        # secondes
    n_frames: int          # nombre de frames estimé
    has_audio: bool


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Exécute une commande et renvoie le résultat (stdout/stderr capturés)."""
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Commande échouée ({proc.returncode}) : {' '.join(cmd)}\n{proc.stderr}"
        )
    return proc


def probe(video_path: str | Path) -> VideoInfo:
    """Lit les métadonnées d'une vidéo avec ffprobe."""
    video_path = str(video_path)
    proc = run([
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", video_path,
    ])
    data = json.loads(proc.stdout)

    vstream = next((s for s in data["streams"] if s.get("codec_type") == "video"), None)
    if vstream is None:
        raise RuntimeError("Aucun flux vidéo trouvé dans le fichier.")
    has_audio = any(s.get("codec_type") == "audio" for s in data["streams"])

    width = int(vstream["width"])
    height = int(vstream["height"])

    # fps : avg_frame_rate est de la forme "30000/1001"
    fps = _parse_fraction(vstream.get("avg_frame_rate") or vstream.get("r_frame_rate") or "0/1")
    if fps <= 0:
        fps = 30.0  # valeur de repli raisonnable

    duration = float(data.get("format", {}).get("duration") or vstream.get("duration") or 0.0)

    n_frames = int(vstream.get("nb_frames") or 0)
    if n_frames <= 0 and duration > 0:
        n_frames = int(round(duration * fps))

    return VideoInfo(width, height, fps, duration, n_frames, has_audio)


def _parse_fraction(frac: str) -> float:
    try:
        num, den = frac.split("/")
        den_f = float(den)
        return float(num) / den_f if den_f != 0 else 0.0
    except (ValueError, ZeroDivisionError):
        try:
            return float(frac)
        except ValueError:
            return 0.0


def reset_dir(path: str | Path) -> Path:
    """Vide (ou crée) un dossier et le renvoie."""
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_frames(folder: str | Path) -> list[Path]:
    """Liste triée des images (png/jpg) d'un dossier."""
    folder = Path(folder)
    files = [p for p in folder.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    return sorted(files)
