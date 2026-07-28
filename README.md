# ScentML: Molecular Olfaction Prediction Engine

**ScentML** is a machine learning system that predicts how molecules smell and recommends fragrance ingredients for a target scent profile. Built on expert-labeled olfactory data, it bridges computational chemistry and fragrance formulation.

---

## What it does

**Forward prediction:** Input any molecule (as a SMILES string) → get a ranked odor profile across 154 scent descriptors (floral, woody, citrus, musk, etc.)

**Reverse recommendation:** Describe a target scent (e.g. "rose + powdery + fresh") → get the top matching molecules from a database of 3,755 compounds

---

## Live Demo

 [Try ScentML on Streamlit](https://scentml-72yfjyzq8zgnenwp7gbjjo.streamlit.app/)

---

## How it works

1. **Data** — GoodScents olfactory dataset via [Pyrfume](https://pyrfume.org): 3,755 molecules with expert odor labels
2. **Molecular features** — Morgan Fingerprints (ECFP4, 2048-bit) generated with RDKit
3. **Model** — Multi-label XGBoost classifier (one estimator per odor descriptor), threshold-tuned for best F1
4. **Explainability** — SHAP TreeExplainer for per-prediction feature importance
5. **Reverse search** — Cosine similarity between a query odor vector and all molecule profiles in the database

---

## Model Performance

| Metric | Score |
|---|---|
| Mean AUC (across 154 labels) | **0.826** |
| F1 Micro (threshold = 0.30) | 0.378 |
| F1 Macro | 0.216 |
| Molecules in database | 3,755 |
| Odor descriptors | 154 |

---

## Tech Stack

- **Python** — RDKit, XGBoost, scikit-learn, SHAP, pandas, numpy
- **App** — Streamlit

---


## Repo Structure
```
ScentML/
├── scent.py # Streamlit app
├── requirements.txt # Dependencies
├── ScentML.ipynb # Full modeling notebook
└── README.md
```
## Run locally

```bash
git clone https://github.com/sjonnavithula09/ScentML.git
cd ScentML
pip install -r requirements.txt
streamlit run scent.py
```

> Note: The trained model (~150MB) downloads automatically on first run.

---

## Background

Olfaction is one of the least digitized senses. Most fragrance development still relies on expert perfumers and manual trial-and-error. ScentML demonstrates how ML can accelerate ingredient discovery by learning the relationship between molecular structure and perceived smell — a core challenge at companies like Osmo, IFF, Givaudan, and Symrise.

---

