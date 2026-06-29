"""
Interface Gradio de WatermarkCleaner.

L'utilisateur :
  1. glisse-dépose une vidéo,
  2. voit la première frame, dessine (au pinceau) la zone du filigrane,
     (option "filigrane mobile" -> dessine aussi sur la dernière frame),
  3. lance le traitement (barre de progression),
  4. voit un aperçu avant/après et télécharge la vidéo nettoyée.

Lancement : python app.py  ->  lien public Gradio (share=True).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import gradio as gr
import numpy as np

# --------------------------------------------------------------------------- #
# Correctif d'un bug connu de gradio_client sur Colab :
# json_schema_to_python_type plante ("argument of type 'bool' is not iterable")
# quand un schéma JSON vaut True/False au lieu d'un dict. On court-circuite ce cas.
# --------------------------------------------------------------------------- #
import gradio_client.utils as _gcu

_orig_j2pt = _gcu._json_schema_to_python_type
_orig_get_type = _gcu.get_type


def _safe_j2pt(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _orig_j2pt(schema, defs)


def _safe_get_type(schema):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_get_type(schema)


_gcu._json_schema_to_python_type = _safe_j2pt
_gcu.get_type = _safe_get_type
# --------------------------------------------------------------------------- #

import config
from pipeline import extract, mask, inpaint, assemble


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _read_rgb(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise gr.Error(f"Impossible de lire l'image {path}.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _editor_value(image_rgb: np.ndarray) -> dict:
    """Construit une valeur de gr.ImageEditor avec l'image en fond."""
    return {"background": image_rgb, "layers": [], "composite": image_rgb}


def _bbox_from_editor(value) -> tuple[int, int, int, int] | None:
    """Déduit le rectangle (x, y, w, h) de la zone peinte dans gr.ImageEditor."""
    if not isinstance(value, dict):
        return None

    mask_bool = None
    for layer in value.get("layers") or []:
        arr = np.asarray(layer)
        if arr.ndim == 3 and arr.shape[2] == 4:
            a = arr[:, :, 3] > 10
        elif arr.ndim == 3:
            a = np.any(arr[:, :, :3] > 10, axis=2)
        else:
            a = arr > 10
        mask_bool = a if mask_bool is None else (mask_bool | a)

    # Repli : différence entre l'image composite et le fond.
    if mask_bool is None or not mask_bool.any():
        bg, comp = value.get("background"), value.get("composite")
        if bg is not None and comp is not None:
            bg = np.asarray(bg)[:, :, :3].astype(int)
            comp = np.asarray(comp)[:, :, :3].astype(int)
            mask_bool = np.abs(comp - bg).sum(axis=2) > 30

    if mask_bool is None or not mask_bool.any():
        return None

    ys, xs = np.where(mask_bool)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #

def on_upload(video_path):
    """À l'upload : extrait première et dernière frames, prépare les éditeurs."""
    if not video_path:
        return None, None, "Aucune vidéo chargée."
    config.ensure_dirs()
    first = config.WORK_DIR / "first_frame.png"
    last = config.WORK_DIR / "last_frame.png"
    extract.extract_first_frame(video_path, first)
    extract.extract_last_frame(video_path, last)
    info = None
    try:
        from pipeline.utils import probe
        info = probe(video_path)
    except Exception:
        pass
    msg = "✅ Vidéo chargée. Dessine la zone du filigrane sur l'image."
    if info:
        msg += f"  ({info.width}x{info.height}, {info.fps:.1f} fps, ~{info.n_frames} frames)"
        w = inpaint.feasibility_warning(info)
        if w:
            msg += "\n\n" + w
    return _editor_value(_read_rgb(first)), _editor_value(_read_rgb(last)), msg


def process(video_path, editor_first, moving, editor_last, progress=gr.Progress()):
    """Pipeline complet, avec barre de progression."""
    if not video_path:
        raise gr.Error("Glisse-dépose une vidéo d'abord.")

    config.ensure_dirs()

    rect0 = _bbox_from_editor(editor_first)
    if rect0 is None:
        raise gr.Error("Dessine la zone du filigrane (au pinceau) sur la première image.")
    rect1 = _bbox_from_editor(editor_last) if moving else None
    if moving and rect1 is None:
        raise gr.Error("« Filigrane mobile » est coché : dessine aussi la zone sur la dernière image.")

    progress(0.05, "Extraction des frames…")
    info = extract.extract_frames(video_path)

    progress(0.15, "Génération des masques…")
    n = mask.generate_masks(rect0, rect1)
    progress(0.2, f"{n} masques générés. Inpainting ProPainter…")

    def pcb(frac, msg):
        progress(min(0.9, 0.2 + frac * 0.65), msg)

    out_video = inpaint.run_propainter(info, pcb)

    progress(0.92, "Réassemblage + audio original…")
    out_name = Path(video_path).stem + "_clean.mp4"
    final = assemble.finalize(out_video, info, out_name)

    # Aperçu avant / après (première frame de chaque).
    before = _read_rgb(config.WORK_DIR / "first_frame.png")
    after_path = config.WORK_DIR / "after_frame.png"
    extract.extract_first_frame(final, after_path)
    after = _read_rgb(after_path)

    progress(1.0, "Terminé ✅")
    status = f"✅ Vidéo nettoyée enregistrée sur Drive : {final}"
    return before, after, final, final, status


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="WatermarkCleaner") as demo:
        gr.Markdown(
            "# 🎬 WatermarkCleaner\n"
            "Supprime un filigrane d'une vidéo (inpainting IA via ProPainter, GPU gratuit Colab).\n\n"
            "**Étapes :** upload → dessine la zone du filigrane au pinceau → lance → télécharge."
        )

        status = gr.Markdown("Charge une vidéo pour commencer.")

        with gr.Row():
            video_in = gr.Video(label="1. Vidéo à nettoyer", sources=["upload"])

        with gr.Row():
            editor_first = gr.ImageEditor(
                label="2. Dessine la zone du filigrane (première frame)",
                type="numpy", brush=gr.Brush(colors=["#FF0000"], default_size=25),
            )
            editor_last = gr.ImageEditor(
                label="Zone sur la dernière frame (filigrane mobile)",
                type="numpy", brush=gr.Brush(colors=["#FF0000"], default_size=25),
                visible=False,
            )

        moving = gr.Checkbox(label="Le filigrane se déplace (interpolation entre 2 positions)", value=False)

        run_btn = gr.Button("3. Lancer le traitement 🚀", variant="primary")

        gr.Markdown("### Aperçu avant / après")
        with gr.Row():
            before_img = gr.Image(label="Avant", type="numpy")
            after_img = gr.Image(label="Après", type="numpy")

        with gr.Row():
            video_out = gr.Video(label="Vidéo nettoyée")
            file_out = gr.File(label="4. Télécharger la vidéo nettoyée")

        # --- Liaisons ---
        video_in.change(
            on_upload, inputs=video_in,
            outputs=[editor_first, editor_last, status],
        )
        moving.change(
            lambda m: gr.update(visible=m), inputs=moving, outputs=editor_last,
        )
        run_btn.click(
            process,
            inputs=[video_in, editor_first, moving, editor_last],
            outputs=[before_img, after_img, video_out, file_out, status],
        )

    return demo


if __name__ == "__main__":
    config.ensure_dirs()
    build_ui().launch(share=True)
