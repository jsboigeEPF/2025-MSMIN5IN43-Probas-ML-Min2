# 2025 - MSMIN5IN43 - IA probabiliste, théorie de jeux et machine learning
# Anaïs DEWEVER - Edwige LEBLANC - Marianne LEPERE

Projet pédagogique d'exploration des approches d'intelligence artificielle probabilistes, de la théorie des jeux et du machine learning pour les étudiants de l'EPF.

---

## 📅 Modalités du projet

### Échéances importantes
- **15 décembre 2025** : Présentation des sujets proposés
- **5 janvier 2026** : Deadline de soumission des projets par Pull Request sur ce dépôt
- **6 janvier 2026** : Présentation finale et rendu

### Date de livraison
Le code avec le README devront être livrés dans un sous-dossier de ce dépôt pour chaque groupe 1 jour au plus tard avant la présentation.

### Taille des groupes
La taille standard d'un groupe est de **3 personnes**.
- Groupes de 2 : toléré (+1 point bonus potentiel pour la charge)
- Groupes de 4 : toléré (-1 point malus potentiel pour la dilution)
- Individuel : exceptionnel (+3 points bonus potentiel)

### Évaluation collégiale
L'évaluation portera sur :
1.  **Présentation/Communication** : Clarté, pédagogie, qualité des slides.
2.  **Contenu théorique** : Compréhension des enjeux, état de l'art, contexte.
3.  **Contenu technique** : Qualité du code, résultats obtenus, démos.
4.  **Organisation/Collaboration** : Activité Git, répartition du travail.

### Livrables attendus
- **Code source** propre et documenté.
- **README** complet (contexte, installation, usage, résultats).
- **Slides** de la présentation (PDF ou lien).

---

### SUJET

### ♟️ Catégorie 2 : Théorie des Jeux & Systèmes Multi-Agents

Ces sujets traitent de la prise de décision stratégique, de la coopération et de la compétition entre agents autonomes.

#### 2.6. Théorie des Jeux appliquée à la Santé & Biologie
La théorie des jeux ne sert pas qu'à jouer, elle modélise le vivant et la société.


- **Sujet B : Théorie des jeux évolutionniste**.
    - Modéliser pourquoi certains comportements (altruisme, agressivité) survivent dans une population.
    - Simuler des dynamiques de type "Hawk-Dove" ou "Rock-Paper-Scissors" dans des populations biologiques.

## STRUCTURE DU PROJET

game-theory/
├─ src/main/java/com/game/gametheory/
│  ├─ model/          # Classes métier : Creature, Hawk, Dove, Board, Species
│  ├─ engine/         # Moteur de jeu : GameEngine, GameSnapshot, CreatureDTO
│  ├─ controller/     # REST API : GameController
│  └─ GameTheoryApplication.java  # Spring Boot main
├─ src/main/resources/static/
│  ├─ game.html       # Frontend
│  └─ game.js         # Logique d’affichage et chart
├─ pom.xml            # Maven configuration
└─ README.md

## INSTALLATION
# Prérequis

- Java 17
- Maven
- IntelliJ IDEA (ou tout IDE compatible Spring Boot)

# Étapes

- Cloner le dépôt :
    git clone <URL_DU_DEPOT>
    cd game-theory
- Ouvrir le projet dans IntelliJ :
    File → Open → game-theory
    IntelliJ détectera le projet Maven et téléchargera les dépendances
- Vérifier le JDK :
    File → Project Structure → Project SDK → sélectionner Java 17

# LANCEMENT

Lancer l’application Spring Boot :
- Ouvrir GameTheoryApplication.java → Run
Ou via Maven :
- mvn spring-boot:run

Accéder au frontend :
- Page d'accueil: http://localhost:8080/index.html
- Simulation Hawks-Dove: http://localhost:8080/HK/hawk-dove.html
- Simulation Rock-Paper-Scissor: http://localhost:8080/RPS/rps.html
- Tableaux des gains: http://localhost:8080/rules.html

## Simulation Hawks-Dove

La page affiche :
- Le plateau: un cercle autour duquel sont placés les créatures (Hawk rouge/Dove bleu/Grudge jaune/Detective violet)
- Un graphique représentant l’évolution des populations
- Des sliders permettant de changer la répartition des créatures entre les catégories avant de commencer la simulation
- Les boutons

Les boutons :
- Start : initialise la simulation
- Stop : met la simulation en pause à la fin du jour en court
- Reset : remet la page dans la configuration de départ

# LOGIQUE DE SIMULATION

Chaque jour les créatures se dirigent aléatoirement vers une paire de nourriture, la nourriture est réparti selon les règles suivantes:
- Hawk ou Dove seul: 2 nourriture
- Dove/Dove: 1 nourriture
- Hawk/Hawk: 1 nourriture mais perte d'énergie à se battre, résultat 0 nourriture
- Hawk/Dove: 0.5 nourriture pour Dove, 1.5 pour Hawk

Les Grudge et les Detectives agissent de manière alernative:
- un Grudge agit comme dove lors d'une première rencontre et enregistre l'id de la créature en face si celle-ci s'est comporté en Hawks afin de se comporter lui aussi en Hawks si jamais ils se recroisent
- un Detective agit comme dove lors d'une première rencontre et enregistre l'id de la créature en face si celle-ci s'est comporté en Dove afin de se comporter en Hawks si jamais ils se recroisent

À la fin de la journée les créatures:
- meurts si elles ont eu 0 nourriture
- on 50% de chance de survie si elles ont eu 0.5 nourriture
- survivent si elles ont eu 1 nourriture
- on 50% de chance de se reproduire si elles ont eu 1.5 nourriture
- se reproduisent si elles ont eu 2 nourriture

## Simulation Rock-Paper-Scissor

La page affiche :
- Le plateau: un cercle autour duquel sont placés les joueurs (Rock rouge/Scissor vert/Paper bleu)
- Un graphique représentant l’évolution des populations
- Des cases permettant de changer la répartition des créatures entre les catégories avant de commencer la simulation (la somme étant toujours égale à 36)
- Les boutons

Les boutons :
- Start : initialise la simulation
- Stop : met la simulation en pause à la fin du jour en court
- Reset : remet la page dans la configuration de départ

# LOGIQUE DE SIMULATION

Chaque jour les joeurs se dirigent aléatoirement vers un point de rencontre:
- Scissor bas Paper
- Paper bas Rock
- Rock bas Scissor

À la fin de la journée les joueurs:
- restent leur type de départ s'ils ont gagnés
- deviennent le type du gagnant s'ils ont perdus

## Tableaux des gains

La page "tableaux des gains" affiche les tableaux des gains pour les deux simulations.