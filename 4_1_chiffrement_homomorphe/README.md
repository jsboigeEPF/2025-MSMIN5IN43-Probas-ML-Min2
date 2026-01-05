# 🔐 Chiffrement homomorphe appliqué au Machine Learning  
## Inférence de risque de crédit sur données chiffrées

**Projet pédagogique – IA probabiliste & Privacy Preserving ML**  
**Sujet 4.1 – Chiffrement Homomorphe**

Réalisé par :  
- Mohammed Amine FARAH  
- Alban FLOUVAT  
- Amine BERRADA  

---

## Contexte et objectif

Dans de nombreux domaines sensibles (finance, santé, assurance), les modèles de Machine Learning doivent traiter des **données à caractère personnel**. Or, le serveur exécutant le modèle ne devrait idéalement **jamais accéder aux données en clair**, afin de garantir la confidentialité et la conformité réglementaire (RGPD).

Ce projet explore l’utilisation du **chiffrement homomorphe (FHE – Fully Homomorphic Encryption)** pour effectuer une **inférence de Machine Learning directement sur des données chiffrées**, sans jamais les déchiffrer côté serveur.

L’objectif est de démontrer qu’il est possible de :
- entraîner un modèle de classification classique,
- le compiler pour le rendre compatible FHE,
- exécuter une prédiction sur un backend "**aveugle**",
- tout en garantissant que **les données et les résultats restent chiffrés côté serveur**.

---

## Cas d’usage étudié

Nous utilisons le dataset **German Credit** (UCI), qui permet de prédire si un client présente un **risque de crédit** (`good` / `bad`) à partir de caractéristiques personnelles et financières.

Ce cas d’usage est représentatif de scénarios réels en finance, où les données sont hautement sensibles.

---

## Approche technique

### Modèle de Machine Learning
- Modèle : **Régression logistique**
- Librairie : **Concrete-ML (Zama)**
- Prétraitement :
  - One-Hot Encoding pour les variables catégorielles
  - Standardisation des variables numériques
- Cible :
  - `good → 0`
  - `bad → 1`

Le modèle est entraîné et évalué **en clair**, puis transformé pour fonctionner en **chiffrement homomorphe**.

---

## Chiffrement homomorphe (FHE)

Le chiffrement homomorphe permet d’effectuer des calculs directement sur des données chiffrées :

- Le **client chiffre les données**
- Le **serveur effectue le calcul sur les données chiffrées**
- Le **résultat reste chiffré**
- Seul le **client peut déchiffrer le résultat**

Le serveur ne voit **jamais** :
- les données d’entrée,
- les résultats,
- la clé privée.

---

## Étapes détaillées du fonctionnement

### 1. Entraînement du modèle (offline)

Le script `src/model.py` réalise les étapes suivantes :

#### a) Chargement et préparation des données
- Chargement du dataset `.arff`
- Vérification explicite du schéma attendu
- Séparation des features (`X`) et de la cible (`y`)
- Mapping explicite `good/bad → 0/1`

#### b) Prétraitement
- Encodage One-Hot des variables catégorielles
- Standardisation des variables numériques
- Le préprocesseur est **appris une seule fois** et sauvegardé

#### c) Entraînement et évaluation en clair
- Entraînement du modèle en clair
- Évaluation via :
  - Accuracy
  - ROC AUC

Ces métriques servent de **référence** avant passage en FHE.

---

### 2. Calibration FHE

Avant de pouvoir chiffrer, le modèle doit être **calibré**.

**Rôle de la calibration**  
La calibration permet :
- d’estimer les **plages de valeurs** manipulées par le modèle,
- de déterminer la **quantification** des nombres réels,
- de fixer les **paramètres cryptographiques** (profondeur, précision).

Elle est effectuée sur un **sous-ensemble représentatif** des données d’entraînement.

> La calibration est indispensable pour garantir que les calculs chiffrés restent corrects et efficaces.

---

### 3. Compilation FHE

Après calibration, le modèle est **compilé** :

**Rôle de la compilation**  
La compilation transforme le modèle de Machine Learning en :
- un **circuit arithmétique** (additions, multiplications),
- compatible avec le chiffrement homomorphe.

Ce circuit est ce que le serveur exécutera sur des données chiffrées.

---

### 4. Génération des artefacts FHE

La compilation génère deux artefacts distincts :
```
artifacts/
└── fhe/
    ├── client.zip
    └── server.zip
```

#### 🔹 `client.zip`
Contient :
- les paramètres cryptographiques côté client
- les éléments nécessaires au chiffrement et au déchiffrement
- **aucune information exploitable par le serveur**

#### 🔹 `server.zip`
Contient :
- le circuit FHE du modèle
- les paramètres nécessaires à l’exécution du calcul chiffré
- **aucune clé privée**

Cette séparation garantit une **séparation stricte des rôles**.

---

### 5. Gestion des clés cryptographiques

Le client génère :
- une **clé privée** (reste strictement côté client),
- des **evaluation keys** (clés publiques auxiliaires).

**Rôle des evaluation keys**  
Elles permettent au serveur :
- d’effectuer des opérations mathématiques sur des données chiffrées,
- sans jamais pouvoir les déchiffrer.

Le serveur reçoit :
- les données chiffrées,
- les evaluation keys,
- mais **jamais la clé privée**.

---

### 6. Backend "aveugle" (serveur)

Le backend (`src/server_api.py`) :
- charge uniquement `server.zip`,
- expose une API FastAPI `/run_fhe`,
- reçoit des données chiffrées,
- exécute le modèle FHE,
- retourne un **résultat chiffré**.

Le backend ne voit **aucune donnée en clair**.

---

### 7. Frontend / Client

Le client (`src/client_app.py`) :
- charge `client.zip`,
- chiffre localement les données utilisateur,
- envoie les données chiffrées au serveur,
- récupère le résultat chiffré,
- déchiffre localement le résultat.

---

## Résultats

### Performances du modèle (en clair)
- ROC AUC ≈ **0.80**
- Accuracy ≈ **0.78**

Ces résultats montrent que le modèle est pertinent pour la tâche.

### Confidentialité
- Le serveur ne reçoit que des **blobs chiffrés**
- Les données et les résultats restent confidentiels
- La clé privée ne quitte jamais le client

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Utilisation

### Entraîner et compiler le modèle
```bash
python -m src.model
```
### Lancer le serveur
```bash
uvicorn src.api:app --reload --port 8000
```
### Lancer le client
```bash
uvicorn src.front:app --reload --port 8500
```
## Conclusion

Ce projet démontre qu’il est possible d’appliquer le chiffrement homomorphe à un cas concret de Machine Learning, en garantissant une confidentialité totale des données.

L’approche reste coûteuse en calcul, mais elle ouvre des perspectives majeures pour :
- la finance,
- la santé,
- les services cloud confidentiels.

Le chiffrement homomorphe constitue une réponse crédible aux enjeux de confidentialité du Machine Learning moderne.