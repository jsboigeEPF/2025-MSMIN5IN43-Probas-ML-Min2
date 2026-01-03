# Projet 3.4 : Résolution de Captcha par Deep Learning

Ce projet implémente une solution de reconnaissance optique de caractères (OCR) spécialisée pour les captchas, utilisant une architecture Deep Learning CRNN (Convolutional Recurrent Neural Network).

## Structure du projet

- `backend/` : Contient l'API (FastAPI) et le code du modèle (PyTorch).
- `frontend/` : Interface web simple pour tester le modèle.
- `data/` : Contient le dataset Kaggle (images et labels).
- `docs/` : Documentation technique détaillée.
- `fonts/` : Les polices d'ecriture utilisees dans la generation des captchas.

## Prérequis

- Python 3.8+
- Un environnement virtuel est recommandé.

## Installation et Lancement

### 1. Installation des dépendances

Ouvrez un terminal à la racine du dossier `3_4_captcha` :

```bash
# Création de l'environnement virtuel (si pas déjà fait)
python3 -m venv .venv

# Activation
source .venv/bin/activate  # Sur Mac/Linux
# .venv\Scripts\activate   # Sur Windows

# Installation
pip install -r requirements.txt
```

### 2. Entraînement du modèle (Optionnel)

Le modèle est déjà entraîné et sauvegardé sous `backend/model.pth`. Si vous souhaitez le réentraîner :

```bash
cd backend
python3 train_model.py
```

### 3. Lancer l'application

Le lancement se fait en deux parties : le serveur API et l'ouverture du fichier HTML.

**Démarrer le Backend :**

```bash
# Depuis le dossier 3_4_captcha/backend
uvicorn main:app --reload
```

Le serveur démarrera sur `http://127.0.0.1:8000`.

**Ouvrir le Frontend :**

Il suffit d'ouvrir le fichier `frontend/index.html` dans votre navigateur web (double-clic sur le fichier).

## Utilisation

1. Sur la page web, cliquez sur le bouton **"Nouveau Test"**.
2. Une image de captcha est tirée aléatoirement du jeu de validation (images que le modèle n'a jamais vues durant l'entraînement).
3. L'image s'affiche.
4. L'application affiche :
   - La **Vraie Valeur** (issue du fichier CSV).
   - La **Prédiction IA** (ce que le modèle a lu).

## Auteurs

Projet réalisé dans le cadre du cours IA Probabiliste & Machine Learning (2025).
