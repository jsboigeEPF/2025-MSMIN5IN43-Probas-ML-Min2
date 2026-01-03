# Interface Graphique - Bayesian Sports Analytics

## Installation

```bash
# S'assurer d'être dans le dossier du projet
cd 1.4-Bayesian-Sports-Analytics

# Activer l'environnement virtuel
source .venv/bin/activate

# Installer Streamlit si nécessaire
pip install streamlit matplotlib seaborn
```

## Lancement de l'application

```bash
# Depuis le dossier racine du projet
cd visual
streamlit run app.py
```

Ou depuis la racine :
```bash
streamlit run visual/app.py
```

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse `http://localhost:8501`

## Fonctionnalités

### Workflow complet en 5 étapes :

1. **Sélection des données**
   - Choisir un championnat parmi 5 ligues disponibles
   - Sélectionner les saisons à analyser
   - Aperçu des données

2. **Préparation des données**
   - Création du mapping équipes → IDs
   - Statistiques sur le dataset
   - Sauvegarde dans `visual/tmp/`

3. **Entraînement du modèle**
   - Configuration des paramètres MCMC (chaînes, warmup, sampling)
   - Entraînement du modèle Stan
   - Diagnostics de convergence (R_hat)
   - Sauvegarde du modèle dans `visual/tmp/`

4. **Analyse des résultats**
   - Visualisation des meilleures attaques
   - Visualisation des meilleures défenses
   - Analyse de l'avantage du terrain
   - Vue d'ensemble Attaque vs Défense

5. **Prédictions**
   - Sélection de 2 équipes (domicile/extérieur)
   - Simulation de 10,000 matchs
   - Probabilités des issues (victoire domicile/nul/victoire extérieure)
   - Top 10 des scores les plus probables
   - Comparaison automatique avec les cotes des bookmakers
   - Détection des paris à valeur

## Gestion du dossier temporaire

- Tous les fichiers générés sont sauvegardés dans `visual/tmp/`
- Le dossier est **réinitialisé** à chaque nouveau workflow
- Bouton "Recommencer" pour nettoyer et repartir de zéro

## Navigation

- Barre latérale pour naviguer entre les étapes
- Les étapes sont débloquées progressivement
- Possibilité de revenir en arrière
- Bouton "Recommencer" pour tout réinitialiser

## Visualisations

- Graphiques en barres pour les forces d'attaque/défense
- Histogramme pour l'avantage du terrain
- Scatter plot Attaque vs Défense
- Graphiques de probabilités pour les prédictions
- Tableaux comparatifs avec bookmakers

## Fichiers générés dans `visual/tmp/`

- `prepared_data.csv` : Données préparées
- `team_mapping.json` : Mapping équipe → ID
- `fit.pkl` : Modèle Stan entraîné

## Structure des chemins

L'application doit être lancée depuis le dossier `visual/` car elle accède à :
- `../data/football_all_leagues.csv` : Données sources
- `../stan/football_model.stan` : Modèle Stan
- `tmp/` : Dossier de travail temporaire

## Recommandations

- Utilisez au moins 2-3 saisons pour avoir suffisamment de données
- Attendez la fin de l'entraînement (peut prendre 2-5 minutes)
- Vérifiez les diagnostics R_hat (doit être < 1.05)
- Environ 1000+ matchs recommandés pour un bon entraînement
