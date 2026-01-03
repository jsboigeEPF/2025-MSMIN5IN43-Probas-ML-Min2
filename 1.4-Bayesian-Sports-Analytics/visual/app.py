import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from cmdstanpy import CmdStanModel

# Configuration de la page
st.set_page_config(
    page_title="Bayesian Sports Analytics",
    page_icon="⚽",
    layout="wide"
)

# Dossier temporaire dans visual/tmp
TMP_DIR = Path("tmp")
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Fonction pour nettoyer le dossier tmp
def clean_tmp():
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(exist_ok=True)


# Session state initialization
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'league_selected' not in st.session_state:
    st.session_state.league_selected = None
if 'model_fitted' not in st.session_state:
    st.session_state.model_fitted = False
if 'data_prepared' not in st.session_state:
    st.session_state.data_prepared = False


# Titre principal
st.title("Bayesian Sports Analytics")
st.markdown("---")

# Sidebar pour navigation
st.sidebar.title("Navigation")
steps = {
    1: "1. Sélection des données",
    2: "2. Préparation des données",
    3: "3. Entraînement du modèle",
    4: "4. Analyse des résultats",
    5: "5. Prédictions"
}

for step_num, step_name in steps.items():
    if st.sidebar.button(step_name, disabled=(step_num > st.session_state.step)):
        st.session_state.step = step_num

st.sidebar.markdown("---")
if st.sidebar.button("Recommencer", type="secondary"):
    clean_tmp()
    st.session_state.step = 1
    st.session_state.league_selected = None
    st.session_state.model_fitted = False
    st.session_state.data_prepared = False
    st.rerun()

# ============================================================
# ÉTAPE 1: SÉLECTION DES DONNÉES
# ============================================================
if st.session_state.step == 1:
    st.header("Étape 1: Sélection des Données")
    
    # Charger les données complètes
    df_all = pd.read_csv("../data/football_all_leagues.csv")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Championnats disponibles")
        leagues = df_all['League'].unique()
        league_counts = df_all.groupby('League').size()
        
        for league in sorted(leagues):
            st.write(f"- {league}: {league_counts[league]} matchs")
    
    with col2:
        st.subheader("Saisons disponibles")
        seasons = sorted(df_all['Season'].unique())
        st.write(f"Saisons: {', '.join(seasons)}")
    
    st.markdown("---")
    
    # Sélection
    col1, col2 = st.columns(2)
    
    with col1:
        selected_league = st.selectbox(
            "Choisir un championnat",
            sorted(leagues),
            index=list(sorted(leagues)).index("Premier League")
        )
    
    with col2:
        available_seasons = sorted(df_all[df_all['League'] == selected_league]['Season'].unique())
        selected_seasons = st.multiselect(
            "Choisir les saisons",
            available_seasons,
            default=available_seasons[-3:] if len(available_seasons) >= 3 else available_seasons
        )
    
    if selected_seasons:
        filtered_df = df_all[
            (df_all['League'] == selected_league) &
            (df_all['Season'].isin(selected_seasons))
        ]
        
        st.info(f"{len(filtered_df)} matchs sélectionnés")
        
        # Preview
        st.subheader("Aperçu des données")
        st.dataframe(filtered_df.head(10), use_container_width=True)
        
        if st.button("Valider et passer à l'étape 2", type="primary"):
            st.session_state.league_selected = selected_league
            st.session_state.selected_seasons = selected_seasons
            st.session_state.filtered_df = filtered_df
            st.session_state.step = 2
            st.rerun()


# ============================================================
# ÉTAPE 2: PRÉPARATION DES DONNÉES
# ============================================================
elif st.session_state.step == 2:
    st.header("Étape 2: Préparation des Données")
    
    if not st.session_state.league_selected:
        st.warning("Veuillez d'abord sélectionner un championnat")
        st.stop()
    
    df = st.session_state.filtered_df.copy()
    
    # Créer mapping équipes
    teams = pd.unique(pd.concat([df["HomeTeam"], df["AwayTeam"]]))
    team2id = {team: i + 1 for i, team in enumerate(sorted(teams))}
    id2team = {i + 1: team for team, i in team2id.items()}
    
    df["home_id"] = df["HomeTeam"].map(team2id)
    df["away_id"] = df["AwayTeam"].map(team2id)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Nombre de matchs", len(df))
        st.metric("Nombre d'équipes", len(teams))
    
    with col2:
        st.metric("Championnat", st.session_state.league_selected)
        st.metric("Saisons", f"{len(st.session_state.selected_seasons)} saisons")
    
    # Afficher les équipes
    st.subheader("Équipes dans le dataset")
    cols = st.columns(4)
    for i, team in enumerate(sorted(teams)):
        cols[i % 4].write(f"- {team}")
    
    # Sauvegarder
    tmp_data_file = TMP_DIR / "prepared_data.csv"
    tmp_mapping_file = TMP_DIR / "team_mapping.json"
    
    df.to_csv(tmp_data_file, index=False)
    with open(tmp_mapping_file, "w") as f:
        json.dump(team2id, f, indent=2)
    
    st.success(f"Données préparées et sauvegardées dans {TMP_DIR}")
    
    if st.button("Passer à l'entraînement du modèle", type="primary"):
        st.session_state.data_prepared = True
        st.session_state.df = df
        st.session_state.team2id = team2id
        st.session_state.id2team = id2team
        st.session_state.step = 3
        st.rerun()


# ============================================================
# ÉTAPE 3: ENTRAÎNEMENT DU MODÈLE
# ============================================================
elif st.session_state.step == 3:
    st.header("Étape 3: Entraînement du Modèle Stan")
    
    if not st.session_state.data_prepared:
        st.warning("Veuillez d'abord préparer les données")
        st.stop()
    
    df = st.session_state.df
    
    # Paramètres MCMC
    st.subheader("Paramètres MCMC")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        chains = st.number_input("Nombre de chaînes", 1, 8, 4)
    with col2:
        warmup = st.number_input("Warmup iterations", 100, 2000, 1000)
    with col3:
        sampling = st.number_input("Sampling iterations", 100, 3000, 2000)
    
    if st.button("Lancer l'entraînement", type="primary"):
        with st.spinner("Entraînement en cours... Cela peut prendre quelques minutes"):
            
            # Préparer les données pour Stan
            stan_data = {
                "N": len(df),
                "T": max(df["home_id"].max(), df["away_id"].max()),
                "home_team": df["home_id"].values,
                "away_team": df["away_id"].values,
                "home_goals": df["HomeGoals"].values.astype(int),
                "away_goals": df["AwayGoals"].values.astype(int),
            }
            
            # Compiler et entraîner
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Compilation du modèle Stan...")
            progress_bar.progress(10)
            
            model = CmdStanModel(stan_file="../stan/football_model.stan")
            
            status_text.text("Échantillonnage MCMC...")
            progress_bar.progress(30)
            
            fit = model.sample(
                data=stan_data,
                chains=chains,
                iter_warmup=warmup,
                iter_sampling=sampling,
                adapt_delta=0.95,
                show_console=False
            )
            
            progress_bar.progress(90)
            status_text.text("Sauvegarde du modèle...")
            
            # Sauvegarder
            tmp_fit_file = TMP_DIR / "fit.pkl"
            with open(tmp_fit_file, "wb") as f:
                pickle.dump(fit, f)
            
            progress_bar.progress(100)
            status_text.text("Entraînement terminé!")
            
            st.session_state.fit = fit
            st.session_state.model_fitted = True
            
        st.success("Modèle entraîné avec succès!")
        
        # Diagnostics
        st.subheader("Diagnostics de convergence")
        summary = fit.summary()
        
        # Afficher R_hat
        r_hat_cols = [c for c in summary.columns if "R_hat" in c]
        if r_hat_cols:
            r_hat_col = r_hat_cols[0]
            max_r_hat = summary[r_hat_col].max()
            
            if max_r_hat < 1.01:
                st.success(f"Excellente convergence (R_hat max = {max_r_hat:.4f})")
            elif max_r_hat < 1.05:
                st.info(f"Convergence acceptable (R_hat max = {max_r_hat:.4f})")
            else:
                st.warning(f"Convergence douteuse (R_hat max = {max_r_hat:.4f})")
        
        with st.expander("Voir le résumé complet"):
            st.dataframe(summary)
        
        # Sauvegarder l'état
        st.session_state.step = 4
        st.session_state.attack = None
        st.session_state.defense = None
        st.session_state.home_adv_mean = None
        
        if st.button("Analyser les résultats", type="primary"):
            st.rerun()


# ============================================================
# ÉTAPE 4: ANALYSE DES RÉSULTATS
# ============================================================
elif st.session_state.step == 4:
    st.header("Étape 4: Analyse des Résultats")
    
    if not st.session_state.model_fitted:
        st.warning("Veuillez d'abord entraîner le modèle")
        st.stop()
    
    fit = st.session_state.fit
    team2id = st.session_state.team2id
    id2team = st.session_state.id2team
    
    # Extraire les paramètres
    posterior = fit.draws_pd()
    
    attack_cols = [c for c in posterior.columns if "attack[" in c]
    defense_cols = [c for c in posterior.columns if "defense[" in c]
    
    attack_mean = posterior[attack_cols].mean().values
    defense_mean = posterior[defense_cols].mean().values
    home_adv = posterior["home_adv"]
    
    # Créer mapping des équipes disponibles (certaines peuvent être absentes)
    n_teams = len(attack_mean)
    id2team_sorted = {}
    team_counter = 0
    for team_id in sorted(id2team.keys()):
        if team_counter < n_teams:
            id2team_sorted[team_counter + 1] = id2team[team_id]
            team_counter += 1
    
    # Créer DataFrame de ranking
    ranking = pd.DataFrame({
        "Team": [id2team_sorted.get(i + 1, f"Team {i+1}") for i in range(len(attack_mean))],
        "Attack": attack_mean,
        "Defense": defense_mean
    })
    
    # Tabs pour différentes visualisations
    tab1, tab2, tab3, tab4 = st.tabs([
        "Attaques", 
        "Défenses", 
        "Home Advantage",
        "Vue d'ensemble"
    ])
    
    with tab1:
        st.subheader("Attaques - Toutes les équipes")
        
        # Curseur pour sélectionner le nombre d'équipes
        n_teams_display = st.slider(
            "Nombre d'équipes à afficher",
            1,
            len(ranking),
            min(10, len(ranking)),
            key="attack_slider"
        )
        
        top_attacks = ranking.sort_values("Attack", ascending=False).head(n_teams_display)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(top_attacks["Team"], top_attacks["Attack"], color='red', alpha=0.7)
            ax.set_xlabel("Force d'attaque")
            ax.set_title(f"Top {n_teams_display} - Meilleures Attaques")
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.dataframe(
                top_attacks[["Team", "Attack"]].reset_index(drop=True),
                use_container_width=True
            )
    
    with tab2:
        st.subheader("Défenses - Toutes les équipes")
        
        # Curseur pour sélectionner le nombre d'équipes
        n_teams_display = st.slider(
            "Nombre d'équipes à afficher",
            1,
            len(ranking),
            min(10, len(ranking)),
            key="defense_slider"
        )
        
        top_defenses = ranking.sort_values("Defense", ascending=True).head(n_teams_display)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(top_defenses["Team"], top_defenses["Defense"], color='blue', alpha=0.7)
            ax.set_xlabel("Force de défense (négatif = meilleur)")
            ax.set_title(f"Top {n_teams_display} - Meilleures Défenses")
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.dataframe(
                top_defenses[["Team", "Defense"]].reset_index(drop=True),
                use_container_width=True
            )
    
    with tab3:
        st.subheader("Avantage du Terrain")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Moyenne (log)", f"{home_adv.mean():.4f}")
            st.metric("Écart-type", f"{home_adv.std():.4f}")
            st.metric("exp(moyenne)", f"{np.exp(home_adv.mean()):.4f}")
            st.metric("% buts supplémentaires", f"{(np.exp(home_adv.mean())-1)*100:.1f}%")
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(home_adv, bins=50, alpha=0.7, color='green', edgecolor='black')
            ax.axvline(home_adv.mean(), color='red', linestyle='--', label='Moyenne')
            ax.set_xlabel("Home Advantage (log-scale)")
            ax.set_ylabel("Fréquence")
            ax.set_title("Distribution de l'avantage du terrain")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
    
    with tab4:
        st.subheader("Vue d'ensemble: Attaque vs Défense")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Scatter plot
        scatter = ax.scatter(
            ranking["Attack"], 
            ranking["Defense"],
            s=100,
            alpha=0.6,
            c=ranking["Attack"] - ranking["Defense"],
            cmap='RdYlGn'
        )
        
        # Ajouter les noms des équipes
        for _, row in ranking.iterrows():
            ax.annotate(
                row["Team"],
                (row["Attack"], row["Defense"]),
                fontsize=8,
                alpha=0.7,
                xytext=(5, 5),
                textcoords='offset points'
            )
        
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.axvline(0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel("Force d'attaque →")
        ax.set_ylabel("← Force de défense (négatif = meilleur)")
        ax.set_title("Attaque vs Défense")
        plt.colorbar(scatter, label="Différence (Attack - Defense)")
        plt.tight_layout()
        st.pyplot(fig)
        
        st.info("""
        **Interprétation:**
        - En haut à droite: Bonnes attaques, mauvaises défenses (jeu offensif)
        - En bas à gauche: Mauvaises attaques, bonnes défenses (jeu défensif)
        - En bas à droite: Bonnes attaques ET bonnes défenses (équipes complètes)
        """)
    
    # Sauvegarder les paramètres
    st.session_state.attack = attack_mean
    st.session_state.defense = defense_mean
    st.session_state.home_adv_mean = home_adv.mean()
    
    if st.button("Faire des prédictions", type="primary"):
        st.session_state.step = 5
        st.rerun()


# ============================================================
# ÉTAPE 5: PRÉDICTIONS
# ============================================================
elif st.session_state.step == 5:
    st.header("Étape 5: Prédictions et Comparaison")
    
    if not st.session_state.model_fitted:
        st.warning("Veuillez d'abord entraîner le modèle")
        st.stop()
    
    team2id = st.session_state.team2id
    attack = st.session_state.attack
    defense = st.session_state.defense
    home_adv = st.session_state.home_adv_mean
    df = st.session_state.df
    
    # Sélection des équipes
    teams = sorted(team2id.keys())
    
    col1, col2 = st.columns(2)
    
    with col1:
        home_team = st.selectbox("Équipe à domicile", teams, key="home")
    
    with col2:
        away_team = st.selectbox("Équipe à l'extérieur", teams, key="away")
    
    if home_team == away_team:
        st.warning("Veuillez sélectionner deux équipes différentes")
        st.stop()
    
    # Fonction de simulation
    def simulate_match(home, away, n=10000):
        h = team2id[home] - 1
        a = team2id[away] - 1
        
        lam_home = np.exp(home_adv + attack[h] - defense[a])
        lam_away = np.exp(attack[a] - defense[h])
        
        gh = np.random.poisson(lam_home, n)
        ga = np.random.poisson(lam_away, n)
        
        return {
            "home_win": (gh > ga).mean(),
            "draw": (gh == ga).mean(),
            "away_win": (gh < ga).mean(),
            "goals_home": gh,
            "goals_away": ga
        }
    
    if st.button("Prédire le match", type="primary"):
        with st.spinner("Simulation en cours..."):
            pred = simulate_match(home_team, away_team)
        
        st.markdown("---")
        st.subheader("Résultats de la prédiction")
        
        # Afficher les probabilités
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                f"Victoire {home_team}",
                f"{pred['home_win']:.1%}",
                f"Cote: {1/pred['home_win']:.2f}"
            )
        
        with col2:
            st.metric(
                "Match nul",
                f"{pred['draw']:.1%}",
                f"Cote: {1/pred['draw']:.2f}"
            )
        
        with col3:
            st.metric(
                f"Victoire {away_team}",
                f"{pred['away_win']:.1%}",
                f"Cote: {1/pred['away_win']:.2f}"
            )
        
        # Graphique des probabilités
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            outcomes = ['Victoire\n' + home_team, 'Nul', 'Victoire\n' + away_team]
            probs = [pred['home_win'], pred['draw'], pred['away_win']]
            colors = ['green', 'gray', 'red']
            
            ax.bar(outcomes, probs, color=colors, alpha=0.7)
            ax.set_ylabel("Probabilité")
            ax.set_title("Probabilités des issues")
            ax.set_ylim(0, 1)
            
            for i, v in enumerate(probs):
                ax.text(i, v + 0.02, f"{v:.1%}", ha='center', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            # Distribution des scores
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # Compter les scores
            scores = {}
            for gh, ga in zip(pred['goals_home'], pred['goals_away']):
                key = f"{gh}-{ga}"
                scores[key] = scores.get(key, 0) + 1
            
            # Top 10 scores
            top_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
            score_labels = [s[0] for s in top_scores]
            score_counts = [s[1]/100 for s in top_scores]
            
            ax.barh(score_labels, score_counts, color='purple', alpha=0.7)
            ax.set_xlabel("Probabilité (%)")
            ax.set_title("Top 10 - Scores les plus probables")
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
        
        # Comparaison avec bookmakers
        st.markdown("---")
        st.subheader("Comparaison avec les Bookmakers")
        
        # Chercher matchs historiques
        matches = df[
            (df["HomeTeam"] == home_team) & 
            (df["AwayTeam"] == away_team)
        ].copy()
        
        if len(matches) > 0:
            # Calculer probas bookmakers
            matches["p_home"] = 1 / matches["OddsHome"]
            matches["p_draw"] = 1 / matches["OddsDraw"]
            matches["p_away"] = 1 / matches["OddsAway"]
            
            # Normalisation
            total = matches["p_home"] + matches["p_draw"] + matches["p_away"]
            matches["p_home"] /= total
            matches["p_draw"] /= total
            matches["p_away"] /= total
            
            # Moyennes
            p_h_book = matches["p_home"].mean()
            p_d_book = matches["p_draw"].mean()
            p_a_book = matches["p_away"].mean()
            
            st.info(f"Basé sur {len(matches)} matchs historiques")
            
            # Tableau comparatif
            comparison = pd.DataFrame({
                "Issue": [f"Victoire {home_team}", "Match nul", f"Victoire {away_team}"],
                "Modèle (%)": [
                    f"{pred['home_win']:.1%}",
                    f"{pred['draw']:.1%}",
                    f"{pred['away_win']:.1%}"
                ],
                "Cote Modèle": [
                    f"{1/pred['home_win']:.2f}",
                    f"{1/pred['draw']:.2f}",
                    f"{1/pred['away_win']:.2f}"
                ],
                "Bookmakers (%)": [
                    f"{p_h_book:.1%}",
                    f"{p_d_book:.1%}",
                    f"{p_a_book:.1%}"
                ],
                "Cote Bookmakers": [
                    f"{1/p_h_book:.2f}",
                    f"{1/p_d_book:.2f}",
                    f"{1/p_a_book:.2f}"
                ],
                "Différence": [
                    f"{pred['home_win'] - p_h_book:+.1%}",
                    f"{pred['draw'] - p_d_book:+.1%}",
                    f"{pred['away_win'] - p_a_book:+.1%}"
                ]
            })
            
            st.dataframe(comparison, use_container_width=True)
            
            # Identifier valeur
            diff_h = pred['home_win'] - p_h_book
            diff_d = pred['draw'] - p_d_book
            diff_a = pred['away_win'] - p_a_book
            
            max_diff = max(abs(diff_h), abs(diff_d), abs(diff_a))
            
            if max_diff > 0.05:
                if abs(diff_h) == max_diff:
                    result = f"Victoire {home_team}" if diff_h > 0 else f"Contre {home_team}"
                elif abs(diff_d) == max_diff:
                    result = "Match nul" if diff_d > 0 else "Contre match nul"
                else:
                    result = f"Victoire {away_team}" if diff_a > 0 else f"Contre {away_team}"
                
                st.success(f"VALEUR POTENTIELLE DÉTECTÉE: {result}")
            else:
                st.info("Pas de différence significative avec les bookmakers")
        else:
            st.warning("Aucun match historique trouvé entre ces équipes pour comparer avec les bookmakers")


# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Bayesian Sports Analytics - MSMIN5IN43</p>
        <p>Modèle hiérarchique bayésien avec Stan</p>
    </div>
    """,
    unsafe_allow_html=True
)
