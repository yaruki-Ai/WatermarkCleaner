"""Étape 2 — Génération des masques par frame (position fixe ou interpolée)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

import config
from .utils import list_frames, reset_dir

# Un rectangle = (x, y, largeur, hauteur) en pixels, dans le repère de la frame.
Rect = Sequence[int]


def _clamp_rect(rect: Rect, w: int, h: int) -> tuple[int, int, int, int]:
    x, y, rw, rh = (int(round(v)) for v in rect)
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    rw = max(1, min(rw, w - x))
    rh = max(1, min(rh, h - y))
    return x, y, rw, rh


def _interp_rect(r0: Rect, r1: Rect, t: float) -> tuple[int, int, int, int]:
    """Interpolation linéaire entre deux rectangles (t dans [0, 1])."""
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(r0, r1))  # type: ignore[return-value]


def _render_mask(width: int, height: int, rect: tuple[int, int, int, int]) -> np.ndarray:
    """Crée un masque binaire (255 dans le rectangle) avec dilatation + flou des bords."""
    mask = np.zeros((height, width), dtype=np.uint8)
    x, y, rw, rh = rect
    cv2.rectangle(mask, (x, y), (x + rw, y + rh), color=255, thickness=-1)

    if config.MASK_DILATION > 0:
        k = config.MASK_DILATION
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        mask = cv2.dilate(mask, kernel)

    if config.MASK_BLUR_KERNEL and config.MASK_BLUR_KERNEL >= 3:
        ksize = config.MASK_BLUR_KERNEL | 1  # force impair
        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
        # Re-binarise après flou pour garder un masque net au centre.
        mask = np.where(mask > 16, 255, 0).astype(np.uint8)

    return mask


# Confiance minimale du suivi (template matching). En dessous, on conserve la
# dernière position connue pour éviter que le masque ne saute n'importe où.
TRACK_CONFIDENCE = 0.30


def _track_positions(frames, r0, width, height):
    """Suit le filigrane sur toutes les frames par template matching.

    Le rectangle peint sur la 1re frame sert de modèle (template). Sur chaque
    frame suivante on cherche où ce modèle réapparaît -> le masque suit le
    filigrane même s'il se déplace partout. Renvoie la liste des rectangles.
    """
    x, y, rw, rh = r0
    first_gray = cv2.cvtColor(cv2.imread(str(frames[0])), cv2.COLOR_BGR2GRAY)
    template = first_gray[y:y + rh, x:x + rw]

    rects = []
    cx, cy = x, y
    for frame_path in frames:
        gray = cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2GRAY)
        # matchTemplate exige template <= image ; garanti par _clamp_rect.
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val >= TRACK_CONFIDENCE:
            cx, cy = max_loc  # (x, y) du coin haut-gauche trouvé
        rects.append(_clamp_rect((cx, cy, rw, rh), width, height))
    return rects


def generate_masks(
    rect_start: Rect,
    rect_end: Optional[Rect] = None,
    track: bool = False,
    frames_dir: str | Path = None,
    masks_dir: str | Path = None,
) -> int:
    """
    Génère un masque PNG par frame.

    - track=False, rect_start seul -> masque fixe (filigrane immobile).
    - track=False, rect_start + rect_end -> interpolation linéaire entre 2 positions.
    - track=True -> suivi automatique du filigrane par template matching
      (gère un filigrane qui se déplace partout, à partir d'un seul dessin).

    Renvoie le nombre de masques générés.
    """
    frames_dir = Path(frames_dir or config.FRAMES_DIR)
    masks_dir = Path(masks_dir or config.MASKS_DIR)

    frames = list_frames(frames_dir)
    if not frames:
        raise RuntimeError(f"Aucune frame trouvée dans {frames_dir}. Lance d'abord l'extraction.")

    sample = cv2.imread(str(frames[0]))
    height, width = sample.shape[:2]

    reset_dir(masks_dir)
    n = len(frames)

    r0 = _clamp_rect(rect_start, width, height)
    r1 = _clamp_rect(rect_end, width, height) if rect_end is not None else None

    tracked = _track_positions(frames, r0, width, height) if track else None

    for i, frame_path in enumerate(frames):
        if tracked is not None:
            rect = tracked[i]
        elif r1 is None:
            rect = r0
        else:
            t = i / max(1, n - 1)
            rect = _clamp_rect(_interp_rect(r0, r1, t), width, height)
        mask = _render_mask(width, height, rect)
        # Même nom de base que la frame -> appariement garanti avec ProPainter.
        out = masks_dir / frame_path.name
        cv2.imwrite(str(out), mask)

    return n
