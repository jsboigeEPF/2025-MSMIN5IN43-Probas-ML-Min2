# src/schema.py
from __future__ import annotations

# --- Colonnes (ordre exact du dataset ARFF, hors target) ---
FEATURES = [
    "checking_status",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment",
    "installment_commitment",
    "personal_status",
    "other_parties",
    "residence_since",
    "property_magnitude",
    "age",
    "other_payment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "own_telephone",
    "foreign_worker",
]

TARGET = "class"

# --- Colonnes catégorielles ---
CAT_COLS = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment",
    "personal_status",
    "other_parties",
    "property_magnitude",
    "other_payment_plans",
    "housing",
    "job",
    "own_telephone",
    "foreign_worker",
]

# --- Colonnes numériques ---
NUM_COLS = [
    "duration",
    "credit_amount",
    "installment_commitment",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents",
]

# --- Domaines catégoriels ---
DOMAINS = {
    "checking_status": ["<0", "0<=X<200", ">=200", "no checking"],
    "credit_history": [
        "no credits/all paid",
        "all paid",
        "existing paid",
        "delayed previously",
        "critical/other existing credit",
    ],
    "purpose": [
        "new car",
        "used car",
        "furniture/equipment",
        "radio/tv",
        "domestic appliance",
        "repairs",
        "education",
        "vacation",
        "retraining",
        "business",
        "other",
    ],
    "savings_status": ["<100", "100<=X<500", "500<=X<1000", ">=1000", "no known savings"],
    "employment": ["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"],
    "personal_status": [
        "male div/sep",
        "female div/dep/mar",
        "male single",
        "male mar/wid",
        "female single",
    ],
    "other_parties": ["none", "co applicant", "guarantor"],
    "property_magnitude": ["real estate", "life insurance", "car", "no known property"],
    "other_payment_plans": ["bank", "stores", "none"],
    "housing": ["rent", "own", "for free"],
    "job": ["unemp/unskilled non res", "unskilled resident", "skilled", "high qualif/self emp/mgmt"],
    "own_telephone": ["none", "yes"],
    "foreign_worker": ["yes", "no"],
}

# --- Mapping du label ---
LABEL_MAP = {"good": 0, "bad": 1}
