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
from pipeline import extract, mask, detect, inpaint, enhance, assemble


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _read_rgb(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise gr.Error(f"Impossible de lire l'image {path}.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _fit_display(image_rgb: np.ndarray, max_h: int = 500) -> np.ndarray:
    """Réduit l'image pour qu'elle tienne entière dans l'éditeur (sans déformer).

    Les vidéos verticales (portrait) sont trop hautes pour la zone de dessin :
    on limite la hauteur d'affichage. Les coordonnées dessinées sont ensuite
    remises à l'échelle de la pleine résolution dans `process`.
    """
    h, w = image_rgb.shape[:2]
    if h > max_h:
        s = max_h / h
        image_rgb = cv2.resize(image_rgb, (max(1, int(w * s)), max_h),
                               interpolation=cv2.INTER_AREA)
    return image_rgb


def _editor_value(image_rgb: np.ndarray) -> dict:
    """Construit une valeur de gr.ImageEditor avec l'image en fond."""
    return {"background": image_rgb, "layers": [], "composite": image_rgb}


def _editor_bg_shape(value):
    """Forme (h, w) de l'image affichée dans l'éditeur, ou None."""
    if not isinstance(value, dict):
        return None
    bg = value.get("background")
    return None if bg is None else np.asarray(bg).shape


def _scale_rect(rect, from_shape, to_shape):
    """Remet un rectangle de l'échelle d'affichage à la pleine résolution."""
    if rect is None or from_shape is None or to_shape is None:
        return rect
    fh, fw = from_shape[0], from_shape[1]
    th, tw = to_shape[0], to_shape[1]
    sx, sy = tw / fw, th / fh
    x, y, w, h = rect
    return (int(round(x * sx)), int(round(y * sy)),
            int(round(w * sx)), int(round(h * sy)))


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
    """À l'upload : extrait la première frame et la prépare pour le dessin."""
    if not video_path:
        return None, "Aucune vidéo chargée."
    config.ensure_dirs()
    first = config.WORK_DIR / "first_frame.png"
    extract.extract_first_frame(video_path, first)
    info = None
    try:
        from pipeline.utils import probe
        info = probe(video_path)
    except Exception:
        pass
    msg = "✅ Vidéo chargée. Peins (au pinceau) par-dessus le filigrane sur l'image."
    if info:
        msg += f"  ({info.width}x{info.height}, {info.fps:.1f} fps, ~{info.n_frames} frames)"
        w = inpaint.feasibility_warning(info)
        if w:
            msg += "\n\n" + w
    return _editor_value(_fit_display(_read_rgb(first))), msg


def process(video_path, auto_detect, sharpen, editor_first, moving, progress=gr.Progress()):
    """Pipeline complet, avec barre de progression."""
    if not video_path:
        raise gr.Error("Glisse-dépose une vidéo d'abord.")

    config.ensure_dirs()

    # En mode manuel, on récupère/recalibre le rectangle AVANT l'extraction.
    rect0 = None
    if not auto_detect:
        rect0 = _bbox_from_editor(editor_first)
        if rect0 is None:
            raise gr.Error("Décoche la détection auto + peins la zone, ou laisse la détection auto cochée.")
        full_first = _read_rgb(config.WORK_DIR / "first_frame.png")
        rect0 = _scale_rect(rect0, _editor_bg_shape(editor_first), full_first.shape)

    progress(0.03, "Extraction des frames…")
    info = extract.extract_frames(video_path)

    if auto_detect:
        progress(0.08, "Détection automatique du texte (OCR)… première fois = téléchargement du modèle.")
        n = detect.detect_text_masks(
            progress=lambda f, m: progress(0.08 + f * 0.17, m))
        detect.release()  # libère la VRAM d'EasyOCR avant ProPainter
    else:
        progress(0.1, "Génération des masques (suivi du filigrane)…" if moving
                 else "Génération des masques…")
        n = mask.generate_masks(rect0, track=bool(moving))
    progress(0.27, f"{n} masques générés. Inpainting par segments…")

    def pcb(frac, msg):
        progress(min(0.82, 0.27 + frac * 0.55), msg)

    frames_dir = inpaint.run_propainter(info, pcb)

    note = ""
    if sharpen:
        # Passe de netteté non-bloquante : si Real-ESRGAN échoue (install, VRAM),
        # on garde quand même la vidéo inpaintée.
        try:
            progress(0.84, "Amélioration de la netteté (Real-ESRGAN)…")
            frames_dir = enhance.enhance_frames(
                frames_dir, progress=lambda f, m: progress(0.84 + f * 0.09, m))
            enhance.release()
        except Exception as e:
            print("⚠️ Real-ESRGAN indisponible, vidéo gardée sans cette passe :", e)
            note = "\n(⚠️ amélioration de netteté ignorée — voir le log de la cellule)"

    progress(0.95, "Réassemblage + audio original…")
    out_name = Path(video_path).stem + "_clean.mp4"
    final = assemble.finalize(frames_dir, info, out_name)

    progress(1.0, "Terminé ✅")
    status = f"✅ Terminé ! Vidéo nettoyée aussi sauvegardée sur ton Drive :\n`{final}`{note}"
    return final, status


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="WatermarkCleaner") as demo:
        gr.Markdown(
            "# 🎬 WatermarkCleaner\n"
            "Supprime les filigranes/texte d'une vidéo (inpainting IA via ProPainter, GPU gratuit Colab).\n\n"
            "**Étapes :** upload → lance (la détection auto repère tout le texte) → télécharge.\n"
            "Traitement par segments en pleine qualité ; le résultat est aussi sauvegardé sur ton Drive."
        )

        status = gr.Markdown("Charge une vidéo pour commencer.")

        with gr.Row():
            video_in = gr.Video(label="1. Vidéo à nettoyer", sources=["upload"])

        auto_detect = gr.Checkbox(
            label="✨ Détecter automatiquement TOUT le texte (recommandé — rien à dessiner)",
            value=True,
            info="Détecte et efface tout texte/filigrane qui apparaît, n'importe où, n'importe quand. "
                 "Décoche seulement si le filigrane est un logo/image (pas du texte).",
        )

        with gr.Group(visible=False) as manual_group:
            editor_first = gr.ImageEditor(
                label="Peins par-dessus le filigrane (mode manuel)",
                type="numpy", brush=gr.Brush(colors=["#FF0000"], default_size=25),
            )
            moving = gr.Checkbox(
                label="Le filigrane se déplace (suivi automatique image par image)",
                value=False,
                info="Coche si le filigrane bouge. Peins-le bien serré pour un meilleur suivi.",
            )

        sharpen = gr.Checkbox(
            label="✨ Améliorer la netteté à la fin (Real-ESRGAN)",
            value=True,
            info="Réduit le flou des zones nettoyées. Plus lent. Décoche si artefacts.",
        )

        run_btn = gr.Button("2. Lancer le traitement 🚀", variant="primary")

        gr.Markdown("### Résultat")
        video_out = gr.Video(label="Vidéo nettoyée (clique pour télécharger)")

        # --- Liaisons ---
        video_in.change(
            on_upload, inputs=video_in,
            outputs=[editor_first, status],
        )
        # Mode manuel visible seulement si la détection auto est décochée.
        auto_detect.change(
            lambda a: gr.update(visible=not a), inputs=auto_detect, outputs=manual_group,
        )
        run_btn.click(
            process,
            inputs=[video_in, auto_detect, sharpen, editor_first, moving],
            outputs=[video_out, status],
        )

    return demo


if __name__ == "__main__":
    config.ensure_dirs()
    build_ui().launch(share=True)
