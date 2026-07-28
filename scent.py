import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import gdown

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics.pairwise import cosine_similarity

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ScentML",
    page_icon="🧴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-box {
        background: rgba(108,99,255,0.1);
        border: 1px solid rgba(108,99,255,0.3);
        border-radius: 12px;
        padding: 14px 10px;
        text-align: center;
        margin-bottom: 8px;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 800;
        color: #a89cff;
        line-height: 1.2;
    }
    .metric-lbl {
        font-size: 0.72rem;
        color: #aaa;
        margin-top: 2px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .tag {
        display: inline-block;
        background: rgba(108,99,255,0.15);
        color: #a89cff;
        border: 1px solid rgba(108,99,255,0.4);
        border-radius: 20px;
        padding: 4px 14px;
        margin: 3px 2px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .info-box {
        background: rgba(255,255,255,0.04);
        border-left: 3px solid #6c63ff;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 0.88rem;
        color: #ccc;
        margin-bottom: 12px;
    }
    .section-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #888;
        margin-bottom: 4px;
    }
    div[data-testid="stTabs"] button {
        font-size: 0.95rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = 'scentml_model.pkl'
GDRIVE_ID  = '1mURWfxwj53oAEJseB5U91kzeTJXXq7dJ'

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model (first run only)..."):
            gdown.download(id=GDRIVE_ID, output=MODEL_PATH, quiet=False)
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

try:
    artifact        = load_model()
    model           = artifact['model']
    selected_labels = artifact['selected_labels']
    db_profiles     = artifact['db_profiles']
    common_ids      = artifact['common_ids']
    cas_to_smiles   = artifact['cas_to_smiles']
    metrics         = artifact['metrics']
except Exception as e:
    st.error(f"Could not load model: {e}. Make sure scentml_model.pkl is in the same folder.")
    st.stop()


# ── Helpers ───────────────────────────────────────────────────────────────────
def smiles_to_morgan(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return gen.GetFingerprintAsNumPy(mol)


def predict_odor_profile(smiles):
    fp = smiles_to_morgan(smiles)
    if fp is None:
        return None
    proba = np.array([
        clf.predict_proba(fp.reshape(1, -1))[0, 1]
        for clf in model.estimators_
    ])
    return pd.Series(proba, index=selected_labels).sort_values(ascending=False)


def mol_to_image(smiles, size=(300, 220)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=size)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf


def recommend_ingredients(target_descriptors, top_n=10):
    query = np.zeros(len(selected_labels))
    valid = {}
    for desc, strength in target_descriptors.items():
        d = desc.lower().strip()
        if d in selected_labels:
            query[selected_labels.index(d)] = float(strength)
            valid[d] = strength
    if not valid:
        return None, []
    scores  = cosine_similarity(query.reshape(1, -1), db_profiles[selected_labels].values)[0]
    top_idx = np.argsort(scores)[::-1][:top_n]
    top_ids = common_ids[top_idx]
    result  = db_profiles.loc[top_ids, list(valid.keys()) + ['smiles']].copy()
    result['Match Score'] = scores[top_idx]
    return result.sort_values('Match Score', ascending=False), list(valid.keys())


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧴 ScentML")
    st.caption("Molecular Olfaction Intelligence")
    st.divider()

    st.markdown("**Model Stats**")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class='metric-box'>
            <div class='metric-val'>0.826</div>
            <div class='metric-lbl'>Accuracy (AUC)</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='metric-box'>
            <div class='metric-val'>{len(selected_labels)}</div>
            <div class='metric-lbl'>Scent Notes</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='metric-box'>
        <div class='metric-val'>{len(common_ids):,}</div>
        <div class='metric-lbl'>Molecules in Database</div>
    </div>""", unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    **How it works**

    ScentML uses machine learning trained on expert-labeled fragrance data to:
    - Predict how any molecule smells
    - Find molecules that match a target scent profile

    **Data source:** GoodScents · [Pyrfume](https://pyrfume.org)
    """)


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("# ScentML")
st.markdown("**Predict how molecules smell — or find the right ingredients for any scent.**")
st.markdown("")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs([
    "🔬  Predict Odor from a Molecule",
    "🎯  Find Ingredients for a Scent"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Forward: Molecule → Odor Profile
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class='info-box'>
        Enter a molecule below to instantly predict its scent profile across 154 odor descriptors.
        Use the quick examples to get started — no chemistry knowledge required.
    </div>
    """, unsafe_allow_html=True)

    examples = {
        "Linalool — found in lavender & coriander": "OC(CCC=C(C)C)(C=C)C",
        "Vanillin — the scent of vanilla":          "COc1cc(C=O)ccc1O",
        "Geraniol — found in rose & geranium":      "CC(=CCC/C(=C/CO)/C)C",
        "Limonene — fresh citrus peel scent":       "CC1=CCC(CC1)C(=C)C",
        "Eugenol — warm clove & spice":             "C=CCc1ccc(O)c(OC)c1",
        "Menthol — cool peppermint":                "CC1CCC(C(C1)O)C(C)C",
    }

    col_ex, col_input = st.columns([1, 2])

    with col_ex:
        st.markdown("<div class='section-label'>Quick examples</div>", unsafe_allow_html=True)
        selected_example = st.radio(
            "", list(examples.keys()), label_visibility="collapsed"
        )

    with col_input:
        smiles_input = st.text_input(
            "Molecule (SMILES format)",
            value=examples[selected_example],
            help="SMILES is a standard text notation for molecules. Pick an example above or paste your own."
        )

        sensitivity = st.slider(
            "Sensitivity — how many scent notes to show",
            min_value=0.1, max_value=0.9,
            value=0.25, step=0.05,
            help="Lower = show more notes. Higher = show only the strongest predicted notes."
        )

    st.divider()

    if smiles_input:
        mol = Chem.MolFromSmiles(smiles_input)
        if mol is None:
            st.error("We couldn't read that molecule. Please check the input or pick an example above.")
        else:
            with st.spinner("Analysing scent profile..."):
                profile = predict_odor_profile(smiles_input)

            col_mol, col_chart = st.columns([1, 2])

            with col_mol:
                st.markdown("**Molecular Structure**")
                img_buf = mol_to_image(smiles_input)
                if img_buf:
                    st.image(img_buf, use_container_width=True)

                st.markdown("**Detected scent notes:**")
                above = profile[profile >= sensitivity]
                if len(above) == 0:
                    st.caption("No notes above current sensitivity. Try lowering the slider.")
                else:
                    tags_html = " ".join([
                        f"<span class='tag'>{l}</span>" for l in above.index
                    ])
                    st.markdown(tags_html, unsafe_allow_html=True)

            with col_chart:
                st.markdown("**Scent probability across all notes (top 20)**")
                top20 = profile.head(20)

                fig, ax = plt.subplots(figsize=(8, 6))
                fig.patch.set_facecolor('#0e1117')
                ax.set_facecolor('#0e1117')

                colors = ['#a89cff' if v >= sensitivity else '#3a3550'
                          for v in top20.values]
                ax.barh(range(len(top20)), top20.values[::-1],
                        color=colors[::-1], alpha=0.95, height=0.7)
                ax.set_yticks(range(len(top20)))
                ax.set_yticklabels(top20.index[::-1], fontsize=10, color='#ddd')
                ax.axvline(sensitivity, color='#ff6b6b', linestyle='--',
                           lw=1.5, label=f'Sensitivity ({sensitivity})')
                ax.set_xlabel('Predicted likelihood', fontsize=10, color='#aaa')
                ax.set_xlim(0, 1)
                ax.tick_params(colors='#aaa')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#333')
                ax.spines['bottom'].set_color('#333')
                ax.legend(fontsize=9, facecolor='#1a1a2e', labelcolor='#ccc')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Reverse: Target Scent → Ingredient Recommendations
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class='info-box'>
        Describe the scent you want to create. Select notes and adjust how strong each one should be.
        ScentML will search 3,755 molecules and return the best matching ingredients.
    </div>
    """, unsafe_allow_html=True)

    presets = {
        "Choose a preset accord...":  {},
        "🌲 Woody Musk":              {"woody": 0.9, "musk": 0.8, "sweet": 0.3},
        "🍋 Fresh Citrus Floral":     {"citrus": 0.9, "fresh": 0.8, "floral": 0.7, "green": 0.5},
        "🌹 Rose Floral":             {"rose": 0.9, "floral": 0.8, "powdery": 0.5, "fruity": 0.3},
        "🍂 Oriental Amber":          {"amber": 0.9, "vanilla": 0.8, "sweet": 0.7, "spicy": 0.5},
        "🌊 Aquatic Marine":          {"marine": 0.9, "fresh": 0.8, "ozone": 0.7, "green": 0.4},
        "☕ Warm Gourmand":           {"vanilla": 0.9, "caramellic": 0.8, "sweet": 0.9, "nutty": 0.5},
    }

    col_preset, col_topn = st.columns([2, 1])
    with col_preset:
        st.markdown("<div class='section-label'>Start from a preset</div>", unsafe_allow_html=True)
        preset_choice = st.selectbox("", list(presets.keys()), label_visibility="collapsed")
    with col_topn:
        st.markdown("<div class='section-label'>Results to show</div>", unsafe_allow_html=True)
        top_n = st.number_input("", min_value=5, max_value=20, value=10,
                                label_visibility="collapsed")

    preset_vals = presets[preset_choice]

    st.markdown("<div class='section-label' style='margin-top:16px'>Select scent notes and set their strength</div>",
                unsafe_allow_html=True)

    selected_descs = st.multiselect(
        "",
        options=selected_labels,
        default=list(preset_vals.keys()) if preset_vals else ["floral", "woody", "musk"],
        label_visibility="collapsed",
        placeholder="Search and add scent notes..."
    )

    target = {}
    if selected_descs:
        cols = st.columns(3)
        for i, desc in enumerate(selected_descs):
            default_val = preset_vals.get(desc, 0.7)
            with cols[i % 3]:
                val = st.slider(
                    f"**{desc.capitalize()}**",
                    0.0, 1.0, float(default_val), 0.05,
                    key=f"slider_{desc}",
                    help=f"Strength of '{desc}' in the target scent (0 = off, 1 = dominant)"
                )
                if val > 0:
                    target[desc] = val

    st.divider()

    if st.button("🔍  Find Matching Ingredients", type="primary", use_container_width=True):
        if not target:
            st.warning("Please add at least one scent note above.")
        else:
            with st.spinner("Searching 3,755 molecules..."):
                results, valid_labels = recommend_ingredients(target, top_n=top_n)

            if results is not None and len(results) > 0:
                st.success(f"Found {len(results)} matching ingredients")

                display_cols = valid_labels + ['Match Score']
                display_df = results[display_cols].copy()
                display_df.index.name = "Molecule ID"

                display_df.columns = [
                    c.title() if c != 'Match Score' else 'Match Score'
                    for c in display_df.columns
                ]

                st.dataframe(
                    display_df.round(3).style
                        .background_gradient(subset=['Match Score'], cmap='Purples')
                        .format("{:.2f}"),
                    use_container_width=True,
                    height=380
                )

                st.markdown("**Top matching molecular structures:**")
                smiles_list = results['smiles'].dropna().head(6).tolist()
                mols = [Chem.MolFromSmiles(s) for s in smiles_list
                        if Chem.MolFromSmiles(s)]

                if mols:
                    scores  = results['Match Score'].head(6).tolist()
                    cas_ids = results.index[:6].tolist()
                    legends = [
                        f"{cas_ids[i]}\nMatch: {scores[i]:.0%}"
                        for i in range(len(mols))
                    ]
                    grid_img = Draw.MolsToGridImage(
                        mols, molsPerRow=3,
                        subImgSize=(320, 240),
                        legends=legends
                    )
                    buf = io.BytesIO()
                    grid_img.save(buf, format='PNG')
                    buf.seek(0)
                    st.image(buf, use_container_width=True)

                csv = display_df.round(3).to_csv()
                st.download_button(
                    "⬇️  Download results as CSV",
                    data=csv,
                    file_name="scentml_ingredients.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.warning("No results found. Try selecting different scent notes.")