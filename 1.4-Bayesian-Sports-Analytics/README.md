# Bayesian Sports Analytics

## 📋 Description du Projet

Ce projet vise à **prédire les résultats sportifs mieux que les bookmakers** en utilisant un modèle bayésien hiérarchique. Le modèle estime la force d'attaque et de défense de chaque équipe dans un championnat de football, tout en prenant en compte l'avantage du terrain (home advantage).

### Objectifs
- Modéliser la force des équipes (attaque/défense) dans la Premier League
- Prendre en compte l'avantage du terrain
- Comparer les prédictions avec les cotes des bookmakers
- Utiliser un modèle hiérarchique bayésien implémenté en **Stan**

### Références
- [Stan Case Studies: Sports](https://mc-stan.org/users/documentation/case-studies.html)
- [Baio & Blangiardo (2010) - Hierarchical model for Serie A](https://discovery.ucl.ac.uk/id/eprint/16040/1/16040.pdf)

---

## 🛠️ Installation des Dépendances

### Prérequis
- Python 3.8+
- pip

### Installation

```bash
# Créer un environnement virtuel (recommandé)
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate  # Windows

# Installer les dépendances
pip install pandas numpy cmdstanpy matplotlib seaborn scipy

# Installer Stan (nécessaire pour cmdstanpy)
python -m cmdstanpy.install_cmdstan
```

### Dépendances Principales
- **pandas** : manipulation des données
- **numpy** : calculs numériques
- **cmdstanpy** : interface Python pour Stan (modèle bayésien)
- **matplotlib, seaborn** : visualisations
- **scipy** : calculs statistiques

---

## 📂 Structure du Projet

```
1.4-Bayesian-Sports-Analytics/
├── README.md                    # Ce fichier
├── data/
│   ├── scrapper.py             # Script de récupération des données
│   ├── football_all_leagues.csv # Données de 5 championnats (5 saisons)
│   ├── premier_league_ready.csv # Données préparées pour le modèle
│   ├── team_mapping.json       # Mapping équipe → ID numérique
│   └── fit.pkl                 # Modèle Stan entraîné (généré)
├── scripts/
│   ├── 01_prepare_data.py      # Préparation des données
│   ├── 02_fit_model.py         # Entraînement du modèle
│   ├── 03_analysis.py          # Analyse des résultats
│   ├── 04_prediction.py        # Prédiction de matchs
│   └── 05_vs_bookmakers.py     # Comparaison avec bookmakers
└── stan/
    └── football_model.stan     # Modèle bayésien hiérarchique
```

---

## Guide d'Utilisation

### Étape 0 : Récupérer les Données (Optionnel)

Les données sont déjà présentes dans `data/football_all_leagues.csv`. Si vous souhaitez les re-télécharger :

```bash
cd data
python scrapper.py
```

Cela télécharge les données des 5 dernières saisons de :
- 🇫🇷 Ligue 1
- 🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League
- 🇪🇸 La Liga
- 🇮🇹 Serie A
- 🇩🇪 Bundesliga

---

### Étape 1 : Préparer les Données

```bash
python scripts/01_prepare_data.py
```

**Ce script :**
- Filtre les données pour la Premier League (saisons 2019-20, 2020-21, 2021-22)
- Crée un mapping équipe → ID numérique
- Génère `premier_league_ready.csv` et `team_mapping.json`

**Sortie :**
```
1140 matchs | 20 équipes
```

---

### Étape 2 : Entraîner le Modèle Stan

```bash
python scripts/02_fit_model.py
```

**Ce script :**
- Charge les données préparées
- Compile et exécute le modèle Stan (`football_model.stan`)
- Utilise 4 chaînes MCMC avec 1000 itérations de warmup et 2000 itérations de sampling
- Sauvegarde le modèle entraîné dans `fit.pkl`

**Durée :** ~1-5 minutes selon la machine

**Sortie :**
- Diagnostics de convergence (R_hat, Eff Sample Size)
- Un bon modèle a R_hat ≈ 1.00 pour tous les paramètres

---

### Étape 3 : Analyser les Résultats

```bash
python scripts/03_analysis.py
```

**Ce script affiche :**
1. **Classement des meilleures attaques** (valeur positive élevée = bonne attaque)
2. **Classement des meilleures défenses** (valeur négative = bonne défense)
3. **Home advantage** (avantage du terrain)

**Exemple de sortie :**
```
============================================================
BEST ATTACKS (Higher = Better)
============================================================
            Team    Attack   Defense
        Man City  0.652003  0.390205
       Liverpool  0.515263  0.337661
         Chelsea  0.328477  0.195838
       Leicester  0.308333  0.032133
...

============================================================
BEST DEFENSES (More Negative = Better)
============================================================
            Team    Attack   Defense
         Norwich -0.546732 -0.335416
       West Brom -0.220977 -0.261192
         Watford -0.263192 -0.235552
           Leeds  0.093898 -0.190058
...

============================================================
HOME ADVANTAGE (log-scale)
============================================================
Mean  : 0.1281
Std   : 0.0352
95% CI: [0.0595, 0.1970]
exp(mean) = 1.1366 → ~13.7% more goals at home
```

**Interprétation :**
- **Attack > 0** → Forte attaque (marque plus de buts que la moyenne)
- **Attack < 0** → Attaque faible (marque moins de buts que la moyenne)
- **Defense > 0** → Défense faible (concède plus de buts que la moyenne)
- **Defense < 0** → Forte défense (concède moins de buts que la moyenne)
- `exp(home_adv) ≈ 1.14` → Jouer à domicile augmente le nombre de buts attendus de ~14%

---

### Étape 4 : Prédire un Match

```bash
python scripts/04_prediction.py
```

**Ce script simule** 10,000 matchs entre deux équipes (ex: Man United vs Chelsea) et calcule :
- Probabilité de victoire domicile
- Probabilité de match nul
- Probabilité de victoire extérieure
- **Cotes équitables** (1/probabilité)

**Exemple de sortie :**
```python
{
    'home_win': 0.4234,
    'Odds_home_win': 2.36,
    'draw': 0.2891,
    'Odds_draw': 3.46,
    'away_win': 0.2875,
    'Odds_away_win': 3.48
}
```

**Modification :** Changez les équipes dans le script :
```python
print(simulate_match("Arsenal", "Tottenham"))
```

---

### Étape 5 : Comparer avec les Bookmakers

```bash
python scripts/05_vs_bookmakers.py
```

**Ce script :**
- Utilise la fonction de prédiction du script 04
- Recherche les matchs historiques entre deux équipes
- Extrait et normalise les cotes des bookmakers
- Compare les prédictions du modèle avec les cotes moyennes
- Identifie automatiquement les paris à valeur (différence >5%)

**Exemple de sortie :**
```
============================================================
Man United vs Chelsea
============================================================
Matchs historiques analyses: 3

MODELE:
  Victoire Man United          : 37.1%  (Cote: 2.70)
  Match nul                    : 26.8%  (Cote: 3.73)
  Victoire Chelsea             : 36.1%  (Cote: 2.77)

BOOKMAKERS (moyenne sur 3 matchs):
  Victoire Man United          : 38.1%  (Cote: 2.62)
  Match nul                    : 27.6%  (Cote: 3.63)
  Victoire Chelsea             : 34.3%  (Cote: 2.92)

DIFFERENCE (Modele - Bookmakers):
  Victoire Man United          : -1.0%
  Match nul                    : -0.8%
  Victoire Chelsea             : +1.8%

=> Pas de difference significative
```

**Modification :** Changez les matchs à analyser dans le script :
```python
compare_match("Liverpool", "Arsenal")
compare_match("Man City", "Chelsea")
```

---

## Modèle Bayésien

### Modèle Hiérarchique

Le modèle (`stan/football_model.stan`) utilise une **distribution de Poisson** pour les buts :

```
home_goals ~ Poisson(λ_home)
away_goals ~ Poisson(λ_away)

log(λ_home) = μ + home_adv + attack[home] - defense[away]
log(λ_away) = μ + attack[away] - defense[home]
```

### Paramètres
- **μ** : Baseline (nombre moyen de buts)
- **home_adv** : Avantage du terrain (log-scale)
- **attack[t]** : Force d'attaque de l'équipe t (valeur positive = bonne attaque)
- **defense[t]** : Force de défense de l'équipe t (valeur négative = bonne défense)

### Priors
- `home_adv ~ Normal(0, 0.5)`
- `attack, defense ~ Normal(0, σ)` avec `σ ~ Exponential(1)`

### Interprétation des Paramètres
- Une équipe avec **attack = +0.5** marque ~65% de buts en plus qu'une équipe moyenne (exp(0.5) ≈ 1.65)
- Une équipe avec **defense = -0.3** concède ~26% de buts en moins qu'une équipe moyenne (exp(-0.3) ≈ 0.74)
- Le **home_adv** d'environ 0.13 signifie ~14% de buts supplémentaires à domicile

---

## Résultats Attendus

1. **Identifier les meilleures équipes** offensives et défensives
2. **Quantifier l'avantage du terrain** (~14% d'augmentation des buts)
3. **Prédire les probabilités** de résultats de matchs
4. **Détecter les paris à valeur** en comparant avec les bookmakers

### Observations Intéressantes
- **Man City et Liverpool** : Excellentes attaques mais défenses faibles (jouent offensivement)
- **Norwich et West Brom** : Excellentes défenses mais attaques catastrophiques (jouent défensivement)
- **Leeds** : Cas intéressant avec une attaque modeste (+0.09) mais une très bonne défense (-0.19)

### Limites
- Le modèle suppose que la forme des équipes est **constante** sur la période
- Ne prend pas en compte les **blessures**, **météo**, **fatigue**
- Les données historiques peuvent ne pas refléter les **changements récents**

---

## Tests & Validation

### Diagnostics MCMC
Vérifiez toujours :
- **R_hat < 1.05** (idéalement < 1.01) → Convergence des chaînes
- **Eff_Sample > 1000** → Échantillons effectifs suffisants

### Validation
- Comparez les prédictions avec les résultats réels (saison suivante)
- Calculez le **Brier Score** ou le **Log Loss**

---

## Personnalisation

### Changer de Championnat
Dans `01_prepare_data.py`, modifiez :
```python
df = df[
    (df["League"] == "La Liga") &  # ← Changez ici
    (df["Season"].isin(["2021-22", "2022-23", "2023-24"]))
].copy()
```


---


