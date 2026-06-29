# 🎬 WatermarkCleaner

Outil **gratuit** de suppression de filigrane vidéo, exécuté sur **Google Colab**
(GPU Nvidia T4 offert), avec une interface **drag & drop** (Gradio). Le moteur
d'inpainting est **[ProPainter](https://github.com/sczhou/ProPainter)** (modèle
pré-entraîné, aucun réentraînement). L'audio d'origine est conservé.

---

## 🚀 Démarrage rapide (Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yaruki-Ai/WatermarkCleaner/blob/main/run_colab.ipynb)

1. Clique sur le badge **« Open in Colab »** ci-dessus.
2. Dans Colab : `Exécution` → `Modifier le type d'exécution` → **GPU (T4)**.
3. Exécute les cellules **une par une, dans l'ordre** :
   1. Vérifier le GPU (`nvidia-smi`)
   2. Cloner le repo
   3. Cloner ProPainter + dépendances + poids
   4. Installer les dépendances
   5. Monter Google Drive
   6. Lancer l'interface → un **lien public Gradio** apparaît
4. Ouvre le lien Gradio, et c'est parti 👇

## 🖱️ Utilisation de l'interface

1. **Glisse-dépose** ta vidéo.
2. La **première frame** s'affiche : **dessine au pinceau** par-dessus le filigrane
   (un simple gribouillage sur la zone suffit, le rectangle est déduit automatiquement).
3. Si le filigrane **se déplace**, coche *« Le filigrane se déplace »* et dessine
   aussi la zone sur la **dernière frame** → la position est interpolée entre les deux.
4. Clique **« Lancer le traitement »** et suis la **barre de progression**.
5. Compare l'**aperçu avant / après**, puis **télécharge** la vidéo nettoyée.
   Une copie est aussi sauvegardée sur ton Drive : `MyDrive/WatermarkCleaner/results/`.

---

## 🧱 Architecture

```
WatermarkCleaner/
├── run_colab.ipynb      # Notebook Colab (cellules étape par étape)
├── app.py               # Interface Gradio
├── config.py            # Chemins + réglages VRAM / masque
├── requirements.txt
├── PLAN.md              # Plan technique détaillé
└── pipeline/
    ├── extract.py       # 1. FFmpeg : vidéo -> frames + audio
    ├── mask.py          # 2. Masques par frame (fixe ou interpolé)
    ├── inpaint.py       # 3. ProPainter (GPU) + chunking VRAM
    ├── assemble.py      # 4. FFmpeg : frames -> vidéo + remux audio
    └── utils.py         # ffprobe, helpers
```

Pipeline : **extraction → masque → inpainting → réassemblage**.
Détails complets dans [PLAN.md](PLAN.md).

---

## ⚠️ Limites du tier gratuit Colab

- Session max **~12 h**, déconnexion après **~90 min d'inactivité**.
- VRAM T4 **~15 Go** : les vidéos longues / HD sont découpées en *chunks*
  automatiquement. La **4K est déconseillée** (risque de dépassement mémoire).
- GPU non garanti (quota gratuit) — réessaie plus tard s'il est indisponible.
- Les résultats sont écrits **progressivement sur Google Drive** → rien n'est perdu
  en cas de coupure.

## 🧩 Dépannage

| Problème | Solution |
|----------|----------|
| Pas de GPU détecté | `Exécution` → `Modifier le type d'exécution` → GPU |
| `ProPainter introuvable` | Relance la **cellule 3** du notebook |
| Out-of-memory (OOM) | Réduis la résolution (1080p/720p) ou raccourcis la vidéo |
| Rien n'est dessiné | Peins bien **par-dessus** le filigrane avant de lancer |

## 📜 Crédits & licence

- Inpainting : [ProPainter](https://github.com/sczhou/ProPainter) (S. Zhou et al.).
- Cet outil est destiné à un usage **personnel et légal** (retirer un filigrane
  d'une vidéo dont tu détiens les droits). Respecte les droits d'auteur.
