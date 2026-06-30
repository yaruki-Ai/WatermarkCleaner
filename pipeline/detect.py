"""
Détection automatique du texte (filigranes) par OCR -> un masque par frame.

On utilise EasyOCR (GPU) pour repérer toutes les zones de texte sur chaque
image. L'utilisateur n'a donc rien à dessiner : tout texte qui apparaît, à
n'importe quel moment, est masqué automatiquement puis effacé par ProPainter.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence

import cv2
import numpy as np

import config
from .utils import list_frames, reset_dir

ProgressFn = Callable[[float, str], None]

_reader = None  # EasyOCR est lourd à initialiser -> on le garde en cache.


def _get_reader(langs: Sequence[str]):
    global _reader
    if _reader is None:
        import easyocr  # import tardif : seulement quand on en a besoin
        try:
            _reader = easyocr.Reader(list(langs), gpu=True)
        except Exception:
            # Repli CPU si le GPU n'est pas dispo (plus lent mais fonctionne).
            _reader = easyocr.Reader(list(langs), gpu=False)
    return _reader


def release() -> None:
    """Libère la mémoire GPU prise par EasyOCR (avant de lancer ProPainter)."""
    global _reader
    _reader = None
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _enhance(img: np.ndarray) -> np.ndarray:
    """Renforce le contraste (CLAHE) pour révéler les filigranes peu visibles."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def _read_all(reader, img: np.ndarray):
    """Détections OCR sur l'image normale + (option) sa version contrastée."""
    results = []
    try:
        results += reader.readtext(img)
    except Exception:
        pass
    if config.OCR_DOUBLE_PASS:
        try:
            results += reader.readtext(_enhance(img))
        except Exception:
            pass
    return results


def _refine_mask(mask: np.ndarray) -> np.ndarray:
    """Dilatation + flou des bords, comme pour les masques manuels."""
    if config.MASK_DILATION > 0:
        k = config.MASK_DILATION
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        mask = cv2.dilate(mask, kernel)
    if config.MASK_BLUR_KERNEL and config.MASK_BLUR_KERNEL >= 3:
        ksize = config.MASK_BLUR_KERNEL | 1
        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
        mask = np.where(mask > 16, 255, 0).astype(np.uint8)
    return mask


def detect_text_masks(
    langs: Sequence[str] = None,
    confidence: float = None,
    frames_dir: str | Path = None,
    masks_dir: str | Path = None,
    progress: Optional[ProgressFn] = None,
) -> int:
    """
    Génère un masque par frame couvrant tout le texte détecté.

    Renvoie le nombre de masques générés. Lève une erreur si aucun texte n'est
    détecté sur l'ensemble de la vidéo (rien à nettoyer).
    """
    langs = langs or config.OCR_LANGS
    confidence = config.OCR_CONFIDENCE if confidence is None else confidence
    frames_dir = Path(frames_dir or config.FRAMES_DIR)
    masks_dir = Path(masks_dir or config.MASKS_DIR)

    frames = list_frames(frames_dir)
    if not frames:
        raise RuntimeError(f"Aucune frame trouvée dans {frames_dir}.")

    reader = _get_reader(langs)
    sample = cv2.imread(str(frames[0]))
    height, width = sample.shape[:2]

    reset_dir(masks_dir)
    n = len(frames)
    total_detections = 0

    for i, frame_path in enumerate(frames):
        img = cv2.imread(str(frame_path))
        mask = np.zeros((height, width), dtype=np.uint8)
        results = _read_all(reader, img)
        for box, _text, conf in results:
            if conf < confidence:
                continue
            pts = np.array(box, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], 255)
            total_detections += 1
        mask = _refine_mask(mask)
        cv2.imwrite(str(masks_dir / frame_path.name), mask)

        if progress and (i % 3 == 0 or i == n - 1):
            progress(i / max(1, n - 1), f"Détection du texte {i + 1}/{n}")

    if total_detections == 0:
        raise RuntimeError(
            "Aucun texte détecté dans la vidéo. Si le filigrane n'est pas du "
            "texte (logo, image), décoche la détection auto et dessine la zone."
        )
    return n
