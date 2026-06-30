"""
Passe finale OPTIONNELLE — Real-ESRGAN pour resharper les frames inpaintées.

ProPainter laisse parfois du flou dans les zones reconstruites. Real-ESRGAN
ré-affine/restaure la texture. Ce n'est pas de l'inpainting génératif (il ne
recrée pas le vrai fond), mais ça redonne du piqué. Léger, tient sur un T4.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Callable, Optional

import cv2

import config
from .utils import list_frames, reset_dir

ProgressFn = Callable[[float, str], None]

_upsampler = None
WEIGHTS_URL = ("https://github.com/xinntao/Real-ESRGAN/releases/download/"
               "v0.1.0/RealESRGAN_x4plus.pth")


def _patch_basicsr() -> None:
    """Contourne le bug `torchvision.transforms.functional_tensor` de basicsr.

    Les versions récentes de torchvision ont supprimé ce module ; on installe un
    shim qui réexporte la fonction attendue, AVANT d'importer basicsr/realesrgan.
    """
    name = "torchvision.transforms.functional_tensor"
    if name in sys.modules:
        return
    try:
        __import__(name)
    except Exception:
        import torchvision.transforms.functional as F
        shim = types.ModuleType(name)
        shim.rgb_to_grayscale = F.rgb_to_grayscale
        sys.modules[name] = shim


def _get_upsampler():
    global _upsampler
    if _upsampler is None:
        _patch_basicsr()
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        use_half = config.USE_FP16
        try:
            import torch
            use_half = use_half and torch.cuda.is_available()
        except Exception:
            use_half = False

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)
        _upsampler = RealESRGANer(
            scale=4, model_path=WEIGHTS_URL, model=model,
            tile=512, tile_pad=10, pre_pad=0, half=use_half,
        )
    return _upsampler


def enhance_frames(
    in_dir: str | Path,
    out_dir: str | Path = None,
    outscale: float = None,
    progress: Optional[ProgressFn] = None,
) -> str:
    """Affine chaque frame avec Real-ESRGAN. Renvoie le dossier des frames affinées."""
    outscale = config.ENHANCE_OUTSCALE if outscale is None else outscale
    in_dir = Path(in_dir)
    out_dir = reset_dir(out_dir or (Path(config.WORK_DIR) / "enhanced_frames"))

    frames = list_frames(in_dir)
    if not frames:
        raise RuntimeError(f"Aucune frame à affiner dans {in_dir}.")

    up = _get_upsampler()
    n = len(frames)
    for i, fp in enumerate(frames):
        img = cv2.imread(str(fp))
        try:
            out, _ = up.enhance(img, outscale=outscale)
        except Exception:
            out = img  # en cas de souci sur une frame, on garde l'originale
        cv2.imwrite(str(out_dir / f"frame_{i:06d}.png"), out)
        if progress and (i % 5 == 0 or i == n - 1):
            progress(i / max(1, n - 1), f"Amélioration netteté {i + 1}/{n}")

    return str(out_dir)


def release() -> None:
    """Libère la mémoire GPU de Real-ESRGAN."""
    global _upsampler
    _upsampler = None
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
