# PLAN — WatermarkCleaner Colab

Outil gratuit de suppression de filigrane vidéo, exécuté sur **Google Colab (GPU T4 gratuit)**, avec interface **Gradio** (drag & drop). Moteur d'inpainting : **ProPainter**.

---

## 1. Vue d'ensemble

```
┌─────────────────────────────────────────────────────────────┐
│  Navigateur (toi)                                            │
│   └── Interface Gradio (lien public share=True)             │
└───────────────┬─────────────────────────────────────────────┘
                │  upload vidéo / dessine zone / clique "Lancer"
                ▼
┌─────────────────────────────────────────────────────────────┐
│  Google Colab — runtime GPU T4 (~15 Go VRAM)                │
│                                                             │
│   app.py (Gradio)                                           │
│      │                                                       │
│      ▼                                                       │
│   pipeline/                                                  │
│    1. extract.py   → FFmpeg : vidéo → frames PNG            │
│    2. mask.py      → masque par frame (fixe ou interpolé)   │
│    3. inpaint.py   → ProPainter (GPU) + chunking VRAM       │
│    4. assemble.py  → FFmpeg : frames → vidéo + remux audio  │
│                                                             │
│   Stockage : /content/drive/MyDrive/WatermarkCleaner/      │
│              (Google Drive monté = résultats persistants)   │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Arborescence du repo

```
WatermarkCleaner/
├── run_colab.ipynb          # Notebook principal (cellules étape par étape)
├── app.py                   # Interface Gradio
├── requirements.txt         # Dépendances pip (hors PyTorch déjà sur Colab)
├── README.md                # "Open in Colab", lancement, limites tier gratuit
├── PLAN.md                  # Ce fichier
├── config.py                # Chemins, constantes, réglages VRAM/chunk
└── pipeline/
    ├── __init__.py
    ├── extract.py           # Extraction frames + métadonnées (fps, résolution, durée)
    ├── mask.py              # Génération masques (interpolation linéaire si mobile)
    ├── inpaint.py           # Wrapper ProPainter + découpage en chunks
    ├── assemble.py          # Réassemblage + remux audio original
    └── utils.py             # Helpers : ffprobe, gestion dossiers Drive, logs
```

ProPainter est **cloné séparément** dans la session Colab (pas copié dans le repo) :
`/content/ProPainter` — on réutilise ses poids pré-entraînés, pas de réentraînement.

---

## 3. Pipeline détaillé

### Étape 1 — Extraction (`extract.py`)
- `ffprobe` récupère : fps, largeur, hauteur, durée, présence audio.
- `ffmpeg` extrait toutes les frames en PNG dans `work/frames/`.
- L'audio original est extrait à part (`work/audio.aac`) pour le remux final.

### Étape 2 — Masque (`mask.py`)
- Entrée : 1 ou 2 rectangles `(x, y, w, h)` dessinés dans Gradio.
  - **1 rectangle** → masque fixe, identique sur toutes les frames.
  - **2 rectangles** → filigrane mobile : interpolation **linéaire** de la position
    entre la frame 0 et la dernière frame.
- Chaque masque : rectangle blanc sur fond noir, avec **dilatation** (quelques px)
  + **flou gaussien** léger sur les bords → meilleure reconstruction ProPainter.
- Masques écrits dans `work/masks/` (PNG, 1 par frame, mêmes noms que les frames).

### Étape 3 — Inpainting (`inpaint.py`)
- Appel de ProPainter sur `frames/` + `masks/`.
- **Gestion VRAM (T4 ~15 Go)** : si `nb_frames × résolution` dépasse un seuil,
  découpage en **chunks temporels** (ex. 50–80 frames) avec quelques frames de
  recouvrement pour éviter les ruptures.
  - Taille de chunk calculée automatiquement dans `config.py` selon résolution.
  - Réglages possibles : sous-échantillonnage interne (`--width/--height`) si 4K.
- Si la vidéo est **trop grosse** même au plus petit chunk → message clair à
  l'utilisateur ("réduis la résolution ou la durée"), pas de crash silencieux.
- Frames nettoyées écrites dans `work/output_frames/`.
- Sauvegarde **progressive sur Drive** après chaque chunk (reprise possible).

### Étape 4 — Réassemblage (`assemble.py`)
- `ffmpeg` recompose la vidéo depuis `output_frames/` au fps d'origine.
- **Remux** de l'audio original (`-c:v ... -c:a copy`) → audio conservé.
- Sortie finale : `/content/drive/MyDrive/WatermarkCleaner/results/<nom>_clean.mp4`.

---

## 4. Interface Gradio (`app.py`)

Flux utilisateur :
1. **Upload** vidéo (drag & drop).
2. Affichage de la **première frame** extraite.
3. **Dessin** d'un rectangle sur le filigrane (composant `gr.Image(tool="sketch")`
   ou `ImageEditor`). Bouton optionnel "filigrane mobile" → 2e position.
4. **Lancer le traitement** → `gr.Progress()` (barre de progression par étape/chunk).
5. **Aperçu avant/après** côte à côte (frame ou court clip).
6. **Téléchargement** de la vidéo nettoyée (`gr.File` / `gr.Video`).

Lancement : `demo.launch(share=True)` → lien public temporaire.

---

## 5. Notebook `run_colab.ipynb` (cellules)

| # | Cellule | Rôle |
|---|---------|------|
| 1 | **Vérif GPU** | `nvidia-smi` — confirme le T4 actif |
| 2 | **Clone repo** | clone WatermarkCleaner depuis GitHub |
| 3 | **Clone ProPainter + poids** | clone repo officiel + download checkpoints |
| 4 | **Install deps** | `pip install -r requirements.txt` (sans réinstaller torch) |
| 5 | **Montage Drive** | `google.colab.drive.mount()` + crée dossiers |
| 6 | **Lancement app** | `python app.py` → lien Gradio public |

Chaque cellule est **indépendante et testable** → en cas d'erreur on voit laquelle.

---

## 6. Dépendances (`requirements.txt`)

Déjà fournis par Colab (NE PAS réinstaller) : `torch`, `torchvision`, CUDA, `ffmpeg`.

À installer :
- `gradio`
- `opencv-python-headless`
- `Pillow`
- `numpy`
- + dépendances propres à ProPainter (depuis son `requirements.txt`) :
  `av`, `einops`, `scikit-image`, `imageio`, `imageio-ffmpeg`, `tqdm`, etc.

---

## 7. Limites du tier gratuit (à connaître)

- Session max **~12 h**, déconnexion après **~90 min d'inactivité**.
- VRAM T4 **~15 Go** → vidéos longues/HD découpées en chunks ; 4K déconseillé.
- GPU pas garanti (quota gratuit) — réessayer plus tard si indisponible.
- Résultats sauvegardés **progressivement sur Drive** → rien de perdu si coupure.

---

## 8. Risques / points de vigilance

- **Compatibilité versions** : ProPainter peut exiger une version torch précise ;
  on s'appuie sur celle de Colab et on adapte si conflit (cellule de diagnostic).
- **Qualité d'inpainting** : dépend de la précision de la zone dessinée et du
  fond derrière le filigrane (fond uni = excellent, fond très détaillé = variable).
- **Filigrane mobile non-linéaire** : l'interpolation est linéaire ; si le
  mouvement est complexe, prévoir plusieurs points serait une évolution future.
