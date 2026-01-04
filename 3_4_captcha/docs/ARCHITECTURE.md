# Architecture Technique

## 1. Vue d'ensemble

Le système suit une architecture client-serveur classique :
- **Frontend** : HTML/JS pur. Interroge l'API pour obtenir des échantillons et des prédictions.
- **Backend** : API REST construite avec FastAPI. Elle charge le modèle PyTorch en mémoire et traite les images.
- **Modèle** : Réseau de neurones profond (Deep Learning) de type CRNN.

## 2. Le Modèle : CRNN (Convolutional Recurrent Neural Network)

L'architecture choisie est l'état de l'art pour la reconnaissance de séquences de texte dans des images (OCR) sans segmentation explicite des caractères.

### Composants :

1.  **Extraction de caractéristiques (CNN)**
    - Une série de couches de convolution (`Conv2d`) et de pooling (`MaxPool2d`) réduit l'image d'entrée (200x70 pixels) en une séquence de vecteurs de caractéristiques.
    - Nous utilisons 3 blocs de convolution pour extraire des formes de plus en plus complexes (bords, courbes, lettres partielles).
    - Sortie : Une "carte" de features de taille `(Batch, Channels, Height, Width)`.

2.  **Traitement Séquentiel (RNN - LSTM)**
    - Les features visuelles sont "aplaties" verticalement pour créer une séquence temporelle correspondant à la lecture de gauche à droite.
    - Un **Bi-LSTM** (Bidirectional Long Short-Term Memory) parcourt cette séquence dans les deux sens pour comprendre le contexte (ex: quelle lettre vient après 'A' dans ce contexte graphique ?).

3.  **Décodage (CTC Loss)**
    - La couche finale prédit, pour chaque "pas de temps" (slice de l'image), une probabilité pour chaque caractère de l'alphabet (+ un caractère "blank").
    - La **CTC (Connectionist Temporal Classification)** permet d'entraîner le modèle sans savoir où se trouve exactement chaque lettre. Elle gère l'alignement entre la séquence d'entrée (l'image large) et la séquence de sortie (le texte court).
    - Au décodage, on fusionne les doublons et on retire les "blanks" pour obtenir le mot final.

## 3. Données

- **Dataset** : Images Kaggle Captcha.
- **Prétraitement** :
    - Conversion en niveaux de gris (1 canal).
    - Redimensionnement fixe à 200x70.
    - Normalisation des pixels entre [-1, 1].
- **Split** :
    - 90% Entraînement.
    - 10% Validation (utilisé pour la démo).

## 4. API (FastAPI)

- `GET /test-sample` : Pioche une image au hasard dans `val_split.csv` et la renvoie avec son label dans un header HTTP `X-True-Label`. Cela garantit que l'on teste sur des données "honnêtes" (non vues).
- `POST /predict` : Reçoit une image binaire, la passe dans le pipeline de prétraitement -> modèle -> décodage, et renvoie la chaîne de caractères prédite.
