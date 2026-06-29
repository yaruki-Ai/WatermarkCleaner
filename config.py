"""
Configuration centrale de WatermarkCleaner.

Tous les chemins et réglages (VRAM, taille de chunk, dilatation/flou du masque)
sont définis ici pour être ajustés facilement.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Chemins
# --------------------------------------------------------------------------- #

# Racine du repo (dossier contenant ce fichier).
REPO_DIR = Path(__file__).resolve().parent

# Emplacement de ProPainter cloné dans la session Colab.
# Surclassable via la variable d'environnement PROPAINTER_DIR.
PROPAINTER_DIR = Path(os.environ.get("PROPAINTER_DIR", "/content/ProPainter"))

# Dossier de travail temporaire (frames, masques, sorties intermédiaires).
# Sur Colab c'est /content/wmc_work ; en local on retombe sur ./work.
_default_work = "/content/wmc_work" if Path("/content").exists() else str(REPO_DIR / "work")
WORK_DIR = Path(os.environ.get("WMC_WORK_DIR", _default_work))

# Google Drive monté dans Colab -> stockage persistant des résultats.
# Surclassable via WMC_DRIVE_DIR. En local on retombe sur ./results.
_default_drive = "/content/drive/MyDrive/WatermarkCleaner" if Path("/content").exists() else str(REPO_DIR / "results")
DRIVE_DIR = Path(os.environ.get("WMC_DRIVE_DIR", _default_drive))
RESULTS_DIR = DRIVE_DIR / "results"   # vidéos nettoyées finales
UPLOADS_DIR = DRIVE_DIR / "uploads"   # copies des vidéos uploadées (sécurité)

# Sous-dossiers de travail
FRAMES_DIR = WORK_DIR / "frames"            # frames extraites
MASKS_DIR = WORK_DIR / "masks"              # masques par frame
PROPAINTER_OUT_DIR = WORK_DIR / "pp_out"    # sortie brute de ProPainter
AUDIO_PATH = WORK_DIR / "audio.m4a"         # audio original extrait

# --------------------------------------------------------------------------- #
# Réglages du masque
# --------------------------------------------------------------------------- #

# Dilatation des bords du rectangle (en pixels) : élargit légèrement la zone
# masquée pour bien couvrir les contours du filigrane.
MASK_DILATION = 8

# Flou gaussien appliqué aux bords du masque (noyau impair). 0 = désactivé.
MASK_BLUR_KERNEL = 9

# --------------------------------------------------------------------------- #
# Réglages VRAM / chunking (GPU T4 ~15 Go)
# --------------------------------------------------------------------------- #

# Colab gratuit n'a que ~12 Go de RAM système. ProPainter charge beaucoup de
# frames en mémoire (calcul du flux optique) -> on PLAFONNE la résolution de
# traitement pour éviter que le système ne tue le process (erreur "code -9").
# Le plus grand côté de l'image est ramené à cette taille avant traitement.
MAX_PROCESS_SIDE = 640


def resize_ratio_for(width: int, height: int) -> float:
    """Facteur de réduction pour ramener le plus grand côté à MAX_PROCESS_SIDE."""
    longest = max(width, height)
    if longest <= MAX_PROCESS_SIDE:
        return 1.0
    return round(MAX_PROCESS_SIDE / longest, 4)


# ProPainter traite la vidéo par "sous-vidéos" (subvideo_length frames à la fois).
# Plus la résolution est haute, plus on réduit ce nombre pour tenir dans la VRAM.
# Le seuil est basé sur le nombre de pixels par frame (largeur * hauteur).
def subvideo_length_for(width: int, height: int) -> int:
    """Nombre de frames traitées simultanément par ProPainter selon la résolution."""
    pixels = width * height
    if pixels <= 640 * 480:        # SD
        return 80
    if pixels <= 1280 * 720:       # 720p
        return 60
    if pixels <= 1920 * 1080:      # 1080p
        return 40
    if pixels <= 2560 * 1440:      # 1440p
        return 24
    return 16                      # 4K et au-delà (risqué)

# Au-delà de cette résolution, on prévient l'utilisateur que ça risque l'OOM.
MAX_SAFE_PIXELS = 3840 * 2160      # 4K

# Active le mode demi-précision (fp16) sur GPU pour économiser la VRAM.
USE_FP16 = True


def ensure_dirs() -> None:
    """Crée tous les dossiers nécessaires (travail + Drive)."""
    for d in (WORK_DIR, FRAMES_DIR, MASKS_DIR, PROPAINTER_OUT_DIR,
              DRIVE_DIR, RESULTS_DIR, UPLOADS_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)
