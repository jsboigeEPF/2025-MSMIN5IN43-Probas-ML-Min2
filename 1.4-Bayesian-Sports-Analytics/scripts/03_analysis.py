import pickle
import pandas as pd
import json
import numpy as np

with open("data/fit.pkl", "rb") as f:
    fit = pickle.load(f)

with open("data/team_mapping.json") as f:
    team2id = json.load(f)

id2team = {v: k for k, v in team2id.items()}

posterior = fit.draws_pd()

attack_cols = [c for c in posterior.columns if "attack[" in c]
defense_cols = [c for c in posterior.columns if "defense[" in c]

attack_mean = posterior[attack_cols].mean().values
defense_mean = posterior[defense_cols].mean().values

ranking = pd.DataFrame({
    "Team": [id2team[i + 1] for i in range(len(attack_mean))],
    "Attack": attack_mean,
    "Defense": defense_mean
})

print("="*60)
print("BEST ATTACKS (Higher = Better)")
print("="*60)
print(ranking.sort_values("Attack", ascending=False).to_string(index=False))
print()

print("="*60)
print("BEST DEFENSES (More Negative = Better)")
print("="*60)
print(ranking.sort_values("Defense", ascending=True).to_string(index=False))
print()

print("="*60)
print("HOME ADVANTAGE (log-scale)")
print("="*60)

home_adv = posterior["home_adv"]

print(f"Mean  : {home_adv.mean():.4f}")
print(f"Std   : {home_adv.std():.4f}")
print(f"95% CI: [{np.percentile(home_adv, 2.5):.4f}, {np.percentile(home_adv, 97.5):.4f}]")
print(f"exp(mean) = {np.exp(home_adv.mean()):.4f} → ~{(np.exp(home_adv.mean())-1)*100:.1f}% more goals at home")
print()

print("="*60)
print("INTERPRETATION:")
print("="*60)
print("• Attack > 0  → Strong offense (scores more goals)")
print("• Attack < 0  → Weak offense (scores fewer goals)")
print("• Defense > 0 → Weak defense (concedes more goals)")
print("• Defense < 0 → Strong defense (concedes fewer goals)")
