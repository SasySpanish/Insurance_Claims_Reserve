"""Calcolo Riserva Sinistri
Metodi: Chain Ladder, Bornhuetter-Ferguson, Cape Cod, Average Cost per Claim,
        Case Outstanding Development, Evaluation & Back-test
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import (
    chain_ladder,
    bornhuetter_ferguson,
    cape_cod,
    frequency_severity,
    average_cost_per_claim,
    tabella_riepilogo_riserve,
    build_development_triangle,
)
from utils.riserva_sinistri import (
    validate_triangle,
    compute_factors,
    age_to_age_matrix,
    case_outstanding_development,
    stima_conteggi_da_triangolo,
    backtest,
)

# ── CONFIGURAZIONE PAGINA ──────────────────────────────────────────────────────
st.set_page_config(page_title="Riserva Sinistri", page_icon="https://i.imgur.com/uFTRZup.png", layout="wide")

# ── STILE GLOBALE ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* Gradiente sfondo principale */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a2a3a 0%, #0d3d4a 25%, #0e4d50 50%, #0f5c52 75%, #105a4a 100%);
    min-height: 100vh;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #071e2b 0%, #0a2f3a 100%) !important;
}
[data-testid="stHeader"] {
    background: transparent;
}

/* Font globale */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #e0f2f1;
}

/* Titolo principale */
h1 { 
    font-weight: 600 !important;
    color: #80cbc4 !important;
    letter-spacing: -0.5px;
}
h2, h3 {
    color: #b2dfdb !important;
    font-weight: 500 !important;
}

/* Card / container metric */
[data-testid="metric-container"] {
    background: rgba(14, 77, 80, 0.35);
    border: 1px solid rgba(128, 203, 196, 0.2);
    border-radius: 12px;
    padding: 12px 16px;
    backdrop-filter: blur(6px);
}
[data-testid="metric-container"] label { color: #80cbc4 !important; font-size: 0.78rem; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e0f2f1 !important; font-family: 'DM Mono', monospace; font-size: 1.3rem; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { color: #4db6ac !important; }

/* Tabs */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: rgba(7, 30, 43, 0.6);
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid rgba(128, 203, 196, 0.15);
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #80cbc4;
    font-size: 0.82rem;
    font-weight: 500;
    padding: 6px 14px;
    transition: all 0.2s;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: linear-gradient(135deg, #00897b, #00695c) !important;
    color: #e0f2f1 !important;
    box-shadow: 0 2px 8px rgba(0, 137, 123, 0.4);
}

/* Input / widget */
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"],
[data-testid="stSlider"] {
    background: rgba(7, 30, 43, 0.7) !important;
    border-color: rgba(128, 203, 196, 0.25) !important;
    color: #e0f2f1 !important;
    border-radius: 8px;
}

/* Bottoni */
[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #00897b 0%, #00695c 100%);
    border: none;
    color: white;
    font-weight: 600;
    border-radius: 8px;
    padding: 8px 20px;
    box-shadow: 0 4px 12px rgba(0, 137, 123, 0.35);
    transition: all 0.2s;
}
[data-testid="stButton"] button[kind="primary"]:hover {
    box-shadow: 0 6px 18px rgba(0, 137, 123, 0.5);
    transform: translateY(-1px);
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid rgba(128, 203, 196, 0.2);
}

/* Divider */
hr { border-color: rgba(128, 203, 196, 0.15) !important; }

/* Warning / info / success */
[data-testid="stAlert"] {
    border-radius: 10px;
    border-left: 4px solid #00897b;
    background: rgba(0, 137, 123, 0.12) !important;
    color: #b2dfdb !important;
}

/* Caption */
[data-testid="stCaptionContainer"] { color: #80cbc4 !important; }

/* Radio / checkbox */
[data-testid="stRadio"] label, [data-testid="stCheckbox"] label { color: #b2dfdb !important; }

/* Sezione card-box */
.card-box {
    background: rgba(14, 77, 80, 0.25);
    border: 1px solid rgba(128, 203, 196, 0.18);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
    backdrop-filter: blur(4px);
}
.warning-box {
    background: rgba(255, 160, 0, 0.12);
    border: 1px solid rgba(255, 160, 0, 0.3);
    border-radius: 10px;
    padding: 12px 16px;
    color: #ffcc80;
    margin-bottom: 12px;
}
.diagnostic-box {
    background: rgba(0, 137, 123, 0.1);
    border: 1px solid rgba(128, 203, 196, 0.2);
    border-radius: 10px;
    padding: 14px 18px;
    margin-top: 12px;
}
</style>
""", unsafe_allow_html=True)

# ── COLORI PLOTLY ──────────────────────────────────────────────────────────────
PALETTE = [
    "#4E79A7",  # blu
    "#F28E2B",  # arancione
    "#E15759",  # rosso
    "#76B7B2",  # teal 
    "#59A14F",  # verde
    "#EDC948",  # giallo
    "#B07AA1",  # viola
    "#FF9DA7",  # rosa
    "#9C755F",  # marrone
    "#BAB0AC"   # grigio
]
PLOTLY_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="#e0f2f1",
    font_family="DM Sans",
    xaxis=dict(gridcolor="rgba(128,203,196,0.1)", zerolinecolor="rgba(128,203,196,0.15)"),
    yaxis=dict(gridcolor="rgba(128,203,196,0.1)", zerolinecolor="rgba(128,203,196,0.15)"),
    legend=dict(bgcolor="rgba(7,30,43,0.6)", bordercolor="rgba(128,203,196,0.2)", borderwidth=1),
)


# ── HELPER: costruisce array dal session input ─────────────────────────────────
def build_triangle_from_session(data: list, n: int) -> np.ndarray:
    tri = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i + j < n:
                tri[i, j] = data[i][j]
    return tri


def invalidate_results():
    """Cancella tutti i risultati calcolati quando il triangolo cambia."""
    for key in ["res_cl", "res_bf", "res_cc", "res_acpc", "res_co", "res_bt"]:
        st.session_state.pop(key, None)


def render_triangle_input(key_prefix: str, n: int, anni_label: list,
                           default_data: list, label: str, modalita:str) -> list:
    """Renderizza l'input di un triangolo generico."""
    st.markdown(f"**{label}**")
    header_cols = st.columns([1] + [1] * n)
    header_cols[0].markdown("**Anno \\ Sviluppo**")
    for j in range(n):
        header_cols[j + 1].markdown(f"**+{j}**")

    triangle_data = []
    for i in range(n):
        row_cols = st.columns([1] + [1] * n)
        row_cols[0].markdown(f"**{anni_label[i]}**")
        row = []
        for j in range(n):
            is_future = (i + j >= n)
            if is_future:
                row_cols[j + 1].markdown('<span style="color:#2a5a5a">▫</span>', unsafe_allow_html=True)
                row.append(np.nan)
            else:
                dv = default_data[i][j]
                default_val = float(dv) if dv is not None and not (isinstance(dv, float) and np.isnan(dv)) else 0.0
                val = row_cols[j + 1].number_input(
                    "", min_value=0.0, value=default_val,
                    step=1000.0, format="%.0f",
                    key=f"{key_prefix}_{modalita}_{n}_{i}_{j}",
                    label_visibility="collapsed",
                    disabled=is_future,
                )
                row.append(val)
        triangle_data.append(row)
    return triangle_data


# ── HEADER ────────────────────────────────────────────────────────────────────
st.title(" **⧨** Riserva Sinistri **⧩** ")
st.caption("Stima IBNR · Chain Ladder · Bornhuetter-Ferguson · Cape Cod · Frequency-Severity · Case Outstanding · Evaluation & Backtest")

# ── TABS ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🟢 **Triangolo** ",
    "🟢 **Chain Ladder** ",
    "🟢 **Bornhuetter-Ferguson** ",
    "🟢 **Cape Cod** ",
    "🟢 **Avg Cost/Claim** ",
    "🟢 **Frequency-Severity** ",
    "🟢 **Case Outstanding** ",
    "🟢 **Evaluation** ",
])
tab_input, tab_cl, tab_bf, tab_cc, tab_acpc, tab_fs, tab_co, tab_eval = tabs


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – TRIANGOLO INPUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_input:
    st.subheader("Triangolo dei Pagamenti Cumulati")
    st.markdown("""
    Inserisci i **pagamenti cumulati** per anno di accadimento e anno di sviluppo.
    """)


    c1, c2 = st.columns([1, 3])
    with c1:
        n_anni = st.slider("Dimensione triangolo", 3, 8, 5, on_change=invalidate_results)
        anno_base = st.number_input("Primo anno di accadimento", 2014, 2023, 2019,
                                    on_change=invalidate_results)
        anni_label = [str(int(anno_base) + i) for i in range(n_anni)]

        st.divider()
        modalita = st.radio("Input", ["Manuale", "Preimpostato"],
                            label_visibility="visible")

    with c2:
        DEMO = {
            5: [
                [210_000, 330_000, 352_000, 361_000, 365_000],
                [195_000, 310_000, 340_000, 351_000, np.nan],
                [220_000, 345_000, 370_000, np.nan, np.nan],
                [230_000, 360_000, np.nan, np.nan, np.nan],
                [240_000, np.nan, np.nan, np.nan, np.nan],
            ],
            3: [
                [100_000, 150_000, 160_000],
                [110_000, 160_000, np.nan],
                [120_000, np.nan, np.nan],
            ],
        }


      

    
        if modalita == "Preimpostato" and n_anni in DEMO:
            default_data = DEMO[n_anni]
            
        elif modalita == "Preimpostato":
            rng = np.random.default_rng(42)
            base = rng.integers(150_000, 300_000, n_anni)
            default_data = []
            for i in range(n_anni):
                row = []
                for j in range(n_anni):
                    if i + j >= n_anni:
                        row.append(np.nan)
                    else:
                        cum = float(base[i]) * (1 + 0.55 + sum(
                            [0.12 if k == 1 else 0.04 for k in range(1, j)]
                        )) if j > 0 else float(base[i])
                        row.append(cum)
                default_data.append(row)
        else:
            default_data = [[None] * n_anni for _ in range(n_anni)]


        triangle_data = render_triangle_input(
                 "tri",
                 n_anni,
                 anni_label,
                 default_data,
                 "Pagamenti cumulati (€)",
                 modalita
        )

        if st.button("💾 Salva triangolo", type="primary"):
            tri_array = build_triangle_from_session(triangle_data, n_anni)

            # Validazione
            warns = validate_triangle(tri_array)
            if warns:
                for w in warns:
                    st.markdown(f'<div class="warning-box">⚠️ {w}</div>', unsafe_allow_html=True)
            else:
                st.success("✅ Triangolo valido!")

            st.session_state["triangle"] = tri_array
            st.session_state["n_anni"] = n_anni
            st.session_state["anni_label"] = anni_label
            invalidate_results()

    if "triangle" in st.session_state:
        st.divider()
        st.subheader("Heatmap Triangolo")
        tri_df = pd.DataFrame(
            st.session_state["triangle"],
            index=st.session_state["anni_label"],
            columns=[f"Svil. +{j}" for j in range(st.session_state["n_anni"])]
        )
        fig_heat = go.Figure(go.Heatmap(
            z=tri_df.values,
            x=tri_df.columns.tolist(),
            y=tri_df.index.tolist(),
            colorscale=[[0, "#071e2b"], [0.5, "#00897b"], [1, "#b2dfdb"]],
            text=np.where(np.isnan(tri_df.values), "",
                          np.vectorize(lambda x: f"€{x:,.0f}")(
                              np.where(np.isnan(tri_df.values), 0, tri_df.values))),
            texttemplate="%{text}",
            hovertemplate="Anno %{y} | %{x}: € %{z:,.0f}<extra></extra>",
        ))
        fig_heat.update_layout(title="Triangolo Cumulato dei Pagamenti", height=350, **PLOTLY_LAYOUT)
        st.plotly_chart(fig_heat, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – CHAIN LADDER
# ══════════════════════════════════════════════════════════════════════════════
with tab_cl:
    st.subheader("Δ Chain Ladder Δ")
    st.markdown("""
    Proietta i pagamenti futuri moltiplicando i valori della diagonale attuale
    per i **fattori di sviluppo età-età**. L'assunzione chiave è che lo sviluppo
    futuro replica quello storico.
    """)

    if "triangle" not in st.session_state:
        st.warning("⚠️ Prima inserisci e salva il triangolo nella tab **Triangolo**.")
    else:
        tri = st.session_state["triangle"]
        anni = st.session_state["anni_label"]
        n = st.session_state["n_anni"]

        col_opt1, col_opt2, col_opt3 = st.columns(3)
        tipo_media_cl = col_opt1.selectbox(
            "Tipo di media fattori",
            ["volume", "semplice", "mediana", "ultimi3"],
            format_func=lambda x: {"volume": "Volume-weighted", "semplice": "Semplice",
                                    "mediana": "Mediana", "ultimi3": "Ultimi 3 anni (vol.)"}[x],
            key="tipo_media_cl",
        )
        tail_cl = col_opt2.number_input("Tail factor", min_value=1.0, max_value=2.0,
                                         value=1.0, step=0.005, format="%.4f", key="tail_cl")

        # Calcola fattori preliminari per mostrare la tabella dei link ratio
        factors_auto = compute_factors(tri, tipo_media_cl)
        ldf_mat = age_to_age_matrix(tri)

        st.markdown("**Link Ratio individuali (età-età)**")
        ldf_df = pd.DataFrame(
            ldf_mat,
            index=anni,
            columns=[f"{j}→{j+1}" for j in range(n - 1)],
        )
        st.dataframe(ldf_df.style.format("{:.4f}", na_rep="—"), use_container_width=True)

        st.markdown("**Selezione fattori età-età** (modifica per override manuale)")
        factor_cols = st.columns(n - 1)
        fattori_manuali = []
        for j in range(n - 1):
            f = factor_cols[j].number_input(
                f"{j}→{j+1}",
                min_value=1.0, max_value=5.0,
                value=float(round(factors_auto[j], 5)),
                step=0.001, format="%.4f",
                key=f"f_cl_{j}",
            )
            fattori_manuali.append(f)

        if st.button("▶️ Esegui Chain Ladder", type="primary"):
            res_cl = chain_ladder(tri, tipo_media=tipo_media_cl,
                                   fattori_manuali=np.array(fattori_manuali),
                                   tail_factor=tail_cl)
            st.session_state["res_cl"] = res_cl
            st.session_state["fattori_cl_usati"] = fattori_manuali

        if "res_cl" in st.session_state:
            res = st.session_state["res_cl"]

            k1, k2, k3 = st.columns(3)
            k1.metric("🟢 Riserva Totale IBNR", f"€ {res['riserva_totale']:,.0f}")
            k2.metric("Fattori selezionati", n - 1)
            k3.metric("Tail factor", f"{res['tail_factor']:.4f}")
            st.divider()

            col_f, col_r = st.columns(2)
            with col_f:
                st.markdown("**Fattori età-età selezionati & CDF cumulativi**")
                df_fact = pd.DataFrame({
                    "Intervallo": [f"+{j} → +{j+1}" for j in range(n - 1)],
                    "Fattore sel.": [f"{f:.4f}" for f in res["fattori_sviluppo"]],
                    "CDF to ult.": [f"{res['cdfs'][j]:.4f}" for j in range(n - 1)],
                    "% Sviluppato": [f"{1/res['cdfs'][j]*100:.1f}%" for j in range(n - 1)],
                })
                st.dataframe(df_fact, use_container_width=True, hide_index=True)

            with col_r:
                st.markdown("**Riserve per anno**")
                df_res = pd.DataFrame({
                    "Anno": anni,
                    "Pagato (€)": [f"€ {v:,.0f}" for v in res["pagati_attuali"]],
                    "Ultimato (€)": [f"€ {v:,.0f}" for v in res["ultimati"]],
                    "Riserva (€)": [f"€ {v:,.0f}" for v in res["riserve_per_anno"]],
                })
                st.dataframe(df_res, use_container_width=True, hide_index=True)

            fig = px.bar(x=anni, y=res["riserve_per_anno"].tolist(),
                         labels={"x": "Anno", "y": "Riserva IBNR (€)"},
                         title="Riserva IBNR — Chain Ladder",
                         color_discrete_sequence=[PALETTE[1]])
            fig.update_layout(height=320, **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – BORNHUETTER-FERGUSON
# ══════════════════════════════════════════════════════════════════════════════
with tab_bf:
    st.subheader("Δ Bornhuetter-Ferguson Δ")
    st.markdown("""
    Credibility blend tra chain ladder (peso = % sviluppato) e perdita attesa a priori
    (peso = % non ancora sviluppato). Più stabile del CL per anni immaturi.
    """)

    if "triangle" not in st.session_state:
        st.warning("⚠️ Prima inserisci e salva il triangolo nella tab **Triangolo**.")
    else:
        tri = st.session_state["triangle"]
        anni = st.session_state["anni_label"]
        n = st.session_state["n_anni"]

        tipo_media_bf = st.selectbox(
            "Tipo di media fattori",
            ["volume", "semplice", "mediana", "ultimi3"],
            format_func=lambda x: {"volume": "Volume-weighted", "semplice": "Semplice",
                                    "mediana": "Mediana", "ultimi3": "Ultimi 3 anni"}[x],
            key="tipo_media_bf",
        )
        tail_bf = st.number_input("Tail factor", min_value=1.0, max_value=2.0,
                                   value=1.0, step=0.005, format="%.4f", key="tail_bf")

        st.markdown("**Premi di rischio per anno (€)**")
        cols_p = st.columns(n)
        premi_bf = []
        for i, anno in enumerate(anni):
            p = cols_p[i].number_input(anno, min_value=0.0, value=float(300_000 + i * 10_000),
                                        step=10_000.0, key=f"prem_bf_{i}")
            premi_bf.append(p)

        lr = st.number_input("Loss Ratio atteso a priori (%)", min_value=1.0, max_value=200.0,
                              value=70.0, step=0.5, key="lr_bf") / 100

        if st.button("▶️ Esegui Bornhuetter-Ferguson", type="primary"):
            res_bf = bornhuetter_ferguson(tri, np.array(premi_bf), lr,
                                           tipo_media=tipo_media_bf, tail_factor=tail_bf)
            st.session_state["res_bf"] = res_bf
            st.session_state["premi_bf"] = premi_bf

        if "res_bf" in st.session_state:
            res = st.session_state["res_bf"]

            k1, k2, k3 = st.columns(3)
            k1.metric("🟢 Riserva Totale BF", f"€ {res['riserva_totale']:,.0f}")
            k2.metric("LR a priori", f"{res['loss_ratio_atteso']*100:.1f}%")
            k3.metric("LR Cape Cod (confronto)", f"{res['elr_cape_cod_confronto']*100:.1f}%")

            # Diagnostica divergenza ELR
            div = res["divergenza_elr_pct"]
            if div > 15:
                st.markdown(f"""<div class="warning-box">
                ⚠️ <strong>Divergenza ELR elevata: {div:.1f}%</strong> — 
                L'ELR selezionato ({res['loss_ratio_atteso']*100:.1f}%) diverge 
                significativamente dall'ELR Cape Cod empirico 
                ({res['elr_cape_cod_confronto']*100:.1f}%). 
                Verifica se la scelta a priori è motivata da dati tariffari aggiornati.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="diagnostic-box">
                ✅ ELR a priori e Cape Cod sono coerenti (divergenza {div:.1f}%).
                </div>""", unsafe_allow_html=True)

            st.divider()
            df_bf = pd.DataFrame({
                "Anno": anni,
                "% Sviluppato": [f"{v*100:.1f}%" for v in res["pct_sviluppato"]],
                "Credibilità CL": [f"{v*100:.1f}%" for v in res["pct_sviluppato"]],
                "Perdita a priori (€)": [f"€ {v:,.0f}" for v in res["perdita_apriori"]],
                "Pagato (€)": [f"€ {v:,.0f}" for v in res["pagati_attuali"]],
                "Ultimato BF (€)": [f"€ {v:,.0f}" for v in res["ultimati"]],
                "Riserva BF (€)": [f"€ {v:,.0f}" for v in res["riserve_per_anno"]],
            })
            st.dataframe(df_bf, use_container_width=True, hide_index=True)

            fig = px.bar(x=anni, y=res["riserve_per_anno"].tolist(),
                         labels={"x": "Anno", "y": "Riserva BF (€)"},
                         title="Riserva IBNR — Bornhuetter-Ferguson",
                         color_discrete_sequence=[PALETTE[3]])
            fig.update_layout(height=320, **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – CAPE COD
# ══════════════════════════════════════════════════════════════════════════════
with tab_cc:
    st.subheader("Δ Cape Cod Δ")
    st.markdown("""
    Come il BF, ma l'ELR è **stimato dai dati stessi** invece che selezionato a priori.
    ELR = Σ pagati / Σ (premi × % sviluppato). Rimuove la soggettività nella scelta del LR.
    """)

    if "triangle" not in st.session_state:
        st.warning("⚠️ Prima inserisci e salva il triangolo nella tab **Triangolo**.")
    else:
        tri = st.session_state["triangle"]
        anni = st.session_state["anni_label"]
        n = st.session_state["n_anni"]

        tipo_media_cc = st.selectbox(
            "Tipo di media fattori",
            ["volume", "semplice", "mediana", "ultimi3"],
            format_func=lambda x: {"volume": "Volume-weighted", "semplice": "Semplice",
                                    "mediana": "Mediana", "ultimi3": "Ultimi 3 anni"}[x],
            key="tipo_media_cc",
        )
        tail_cc = st.number_input("Tail factor", min_value=1.0, max_value=2.0,
                                   value=1.0, step=0.005, format="%.4f", key="tail_cc")

        st.markdown("**Premi di rischio per anno (€)** — usati per calcolare l'ELR empirico")
        st.markdown("""<div class="warning-box">
        ⚠️ I premi devono essere corretti per il livello di tariffa (rate-level adjusted).
        Premi non adeguati distorcono l'ELR stimato.
        </div>""", unsafe_allow_html=True)

        cols_cc = st.columns(n)
        premi_cc = []
        for i, anno in enumerate(anni):
            p = cols_cc[i].number_input(anno, min_value=0.0, value=float(300_000 + i * 10_000),
                                         step=10_000.0, key=f"prem_cc_{i}")
            premi_cc.append(p)

        if st.button("▶️ Esegui Cape Cod", type="primary"):
            res_cc = cape_cod(tri, np.array(premi_cc), tipo_media=tipo_media_cc, tail_factor=tail_cc)
            st.session_state["res_cc"] = res_cc
            st.session_state["premi_cc"] = premi_cc

        if "res_cc" in st.session_state:
            res = st.session_state["res_cc"]

            k1, k2 = st.columns(2)
            k1.metric("🟢 Riserva Totale CC", f"€ {res['riserva_totale']:,.0f}")
            k2.metric("ELR Stimato (empirico)", f"{res['elr_stimato']*100:.2f}%")
            st.divider()

            df_cc = pd.DataFrame({
                "Anno": anni,
                "% Sviluppato": [f"{v*100:.1f}%" for v in res["pct_sviluppato"]],
                "Perdita a priori CC (€)": [f"€ {v:,.0f}" for v in res["perdita_apriori"]],
                "Pagato (€)": [f"€ {v:,.0f}" for v in res["pagati_attuali"]],
                "Ultimato CC (€)": [f"€ {v:,.0f}" for v in res["ultimati"]],
                "Riserva CC (€)": [f"€ {v:,.0f}" for v in res["riserve_per_anno"]],
            })
            st.dataframe(df_cc, use_container_width=True, hide_index=True)

            fig = px.bar(x=anni, y=res["riserve_per_anno"].tolist(),
                         labels={"x": "Anno", "y": "Riserva CC (€)"},
                         title="Riserva IBNR — Cape Cod",
                         color_discrete_sequence=[PALETTE[4]])
            fig.update_layout(height=320, **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 – AVERAGE COST PER CLAIM
# ══════════════════════════════════════════════════════════════════════════════
with tab_acpc:
    st.subheader("Δ Average Cost per Claim (ACPC) Δ")
    st.markdown("""
    Riserva = N° sinistri IBNR × Costo Medio (con eventuale inflazione).
    Il numero di sinistri IBNR può essere inserito manualmente o **stimato dal
    triangolo dei conteggi** (chain ladder su claim counts).
    """)

    use_count_triangle = st.checkbox(
        "Stima sinistri IBNR da triangolo dei conteggi (chain ladder su conteggi)",
        value=False,
    )

    n_acpc = st.session_state.get("n_anni", 5)
    anno_base_acpc = st.number_input("Primo anno", 2014, 2023, 2019, key="ab_acpc")
    anni_acpc = [int(anno_base_acpc) + i for i in range(n_acpc)]
    anno_val = st.number_input("Anno di valutazione", 2022, 2030,
                                int(anno_base_acpc) + n_acpc)

    col_a, col_b = st.columns(2)
    costo_medio = col_a.number_input("Costo medio per sinistro (€)", 100.0, 500_000.0,
                                      8_000.0, step=500.0)
    inflazione = col_b.number_input("Inflazione annua (%)", 0.0, 20.0, 3.0, step=0.5)

    n_sin_acpc = []
    if use_count_triangle and "triangle" in st.session_state:
        st.markdown("**Triangolo dei conteggi sinistri (numeri, non valori monetari)**")
        n = st.session_state["n_anni"]
        anni_label = st.session_state["anni_label"]
        default_counts = [[None] * n for _ in range(n)]
        count_data = render_triangle_input("cnt", n, anni_label, default_counts,
                                            "Conteggi cumulati (n° sinistri)", modalita)
        if st.button("📊 Stima IBNR da conteggi", type="primary"):
            cnt_tri = build_triangle_from_session(count_data, n)
            n_sin_acpc_arr = stima_conteggi_da_triangolo(cnt_tri)
            st.session_state["n_sin_acpc_stimati"] = n_sin_acpc_arr
            st.success("Sinistri IBNR stimati dal triangolo dei conteggi!")

        if "n_sin_acpc_stimati" in st.session_state:
            est = st.session_state["n_sin_acpc_stimati"]
            df_est = pd.DataFrame({"Anno": anni_label, "Sinistri IBNR stimati": est})
            st.dataframe(df_est, use_container_width=True, hide_index=True)
            n_sin_acpc = list(est)
    else:
        st.markdown("**Numero sinistri IBNR stimati per anno (inserimento manuale)**")
        cols_acpc = st.columns(n_acpc)
        for i in range(n_acpc):
            val = cols_acpc[i].number_input(str(anni_acpc[i]), min_value=0.0,
                                             value=float(max(0, 50 - i * 5)),
                                             step=1.0, key=f"nacpc_{i}")
            n_sin_acpc.append(val)

    if n_sin_acpc and st.button("▶️ Esegui ACPC", type="primary"):
        res_acpc = average_cost_per_claim(
            np.array(n_sin_acpc), costo_medio, inflazione,
            [int(a) for a in anni_acpc], int(anno_val)
        )
        st.session_state["res_acpc"] = res_acpc

    if "res_acpc" in st.session_state:
        res = st.session_state["res_acpc"]
        k1, k2, k3 = st.columns(3)
        k1.metric("🟢 Riserva Totale ACPC", f"€ {res['riserva_totale']:,.0f}")
        k2.metric("Costo medio base", f"€ {res['costo_medio_base']:,.0f}")
        k3.metric("Inflazione applicata", f"{res['fattore_inflazione_perc']:.1f}%")

        df_acpc = pd.DataFrame({
            "Anno": [str(a) for a in anni_acpc],
            "Sin. IBNR": [f"{v:.0f}" for v in res["numero_sinistri_ibnr"]],
            "Costo Inflazionato (€)": [f"€ {v:,.0f}" for v in res["costi_inflazionati"]],
            "Riserva (€)": [f"€ {v:,.0f}" for v in res["riserve_per_anno"]],
        })
        st.dataframe(df_acpc, use_container_width=True, hide_index=True)

        fig = px.bar(x=[str(a) for a in anni_acpc], y=res["riserve_per_anno"].tolist(),
                     labels={"x": "Anno", "y": "Riserva (€)"},
                     title="Riserva IBNR — Average Cost per Claim",
                     color_discrete_sequence=[PALETTE[7]])
        fig.update_layout(height=320, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)



# ══════════════════════════════════════════════════════════════════════════════
# TAB F-S – FREQUENCY-SEVERITY
# ══════════════════════════════════════════════════════════════════════════════
with tab_fs:
    st.subheader("📈 Frequency-Severity")
    st.markdown("""
    Proietta separatamente **frequenza** (conteggi sinistri) e **severità**
    (costo medio), poi combina: Ultimato = Conteggi ultimati × Severità ultimata.
    Richiede un secondo triangolo con i conteggi cumulati.
    """)

    if "triangle" not in st.session_state:
        st.warning("⚠️ Prima inserisci e salva il triangolo dei pagamenti nella tab **Triangolo**.")
    else:
        tri_paid = st.session_state["triangle"]
        anni = st.session_state["anni_label"]
        n = st.session_state["n_anni"]

        col_fs1, col_fs2, col_fs3 = st.columns(3)
        tipo_media_fs = col_fs1.selectbox(
            "Tipo di media fattori",
            ["volume", "semplice", "mediana", "ultimi3"],
            format_func=lambda x: {"volume": "Volume-weighted", "semplice": "Semplice",
                                    "mediana": "Mediana", "ultimi3": "Ultimi 3 anni"}[x],
            key="tipo_media_fs",
        )
        tail_paid_fs = col_fs2.number_input("Tail factor pagamenti", min_value=1.0,
                                             max_value=2.0, value=1.0, step=0.005,
                                             format="%.4f", key="tail_paid_fs")
        tail_cnt_fs = col_fs3.number_input("Tail factor conteggi", min_value=1.0,
                                            max_value=2.0, value=1.0, step=0.005,
                                            format="%.4f", key="tail_cnt_fs")

        st.markdown("**Triangolo conteggi sinistri cumulati (n° sinistri)**")
        DEMO_COUNTS = {
            5: [
                [120, 175, 188, 193, 195],
                [115, 168, 181, 187, np.nan],
                [125, 180, 194, np.nan, np.nan],
                [130, 188, np.nan, np.nan, np.nan],
                [135, np.nan, np.nan, np.nan, np.nan],
            ],
        }
        default_counts = DEMO_COUNTS.get(n, [[None] * n for _ in range(n)])
        count_data = render_triangle_input("fs_cnt", n, anni, default_counts,
                                            "Conteggi cumulati (n° sinistri)",modalita)

        if st.button("▶️ Esegui Frequency-Severity", type="primary"):
            cnt_array = build_triangle_from_session(count_data, n)
            try:
                res_fs = frequency_severity(tri_paid, cnt_array,
                                             tipo_media=tipo_media_fs,
                                             tail_factor_paid=tail_paid_fs,
                                             tail_factor_counts=tail_cnt_fs)
                st.session_state["res_fs"] = res_fs
            except Exception as e:
                st.error(f"Errore: {e}")

        if "res_fs" in st.session_state:
            res = st.session_state["res_fs"]

            k1, k2, k3 = st.columns(3)
            k1.metric("🟢 Riserva Totale F-S", f"€ {res['riserva_totale']:,.0f}")
            k2.metric("Conteggi ultimati totali", f"{res['ultimati_counts'].sum():,.0f}")
            k3.metric("Severity media ultimata", f"€ {res['ultimati_severity'].mean():,.0f}")

            st.divider()

            # Tabella separata frequenza e severità
            col_f, col_s = st.columns(2)
            with col_f:
                st.markdown("**Proiezione Frequenza (conteggi)**")
                df_freq = pd.DataFrame({
                    "Anno": anni,
                    "Conteggi attuali": [f"{v:.0f}" for v in res["counts_diagonale"]],
                    "Conteggi ultimati": [f"{v:.0f}" for v in res["ultimati_counts"]],
                    "IBNR conteggi": [
                        f"{max(0, u - a):.0f}"
                        for u, a in zip(res["ultimati_counts"], res["counts_diagonale"])
                    ],
                })
                st.dataframe(df_freq, use_container_width=True, hide_index=True)

            with col_s:
                st.markdown("**Proiezione Severità (costo medio)**")
                df_sev = pd.DataFrame({
                    "Anno": anni,
                    "Severity attuale (€)": [f"€ {v:,.0f}" for v in res["severity_diagonale"]],
                    "Severity ultimata (€)": [f"€ {v:,.0f}" for v in res["ultimati_severity"]],
                })
                st.dataframe(df_sev, use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("**Riepilogo F-S per anno**")
            df_fs = pd.DataFrame({
                "Anno": anni,
                "Conteggi ult.": [f"{v:.0f}" for v in res["ultimati_counts"]],
                "Severity ult. (€)": [f"€ {v:,.0f}" for v in res["ultimati_severity"]],
                "Ultimato F-S (€)": [f"€ {v:,.0f}" for v in res["ultimati"]],
                "Pagato attuale (€)": [f"€ {v:,.0f}" for v in res["pagati_attuali"]],
                "Riserva F-S (€)": [f"€ {v:,.0f}" for v in res["riserve_per_anno"]],
            })
            st.dataframe(df_fs, use_container_width=True, hide_index=True)

            fig_fs = go.Figure()
            fig_fs.add_trace(go.Bar(name="Pagato attuale", x=anni,
                                     y=res["pagati_attuali"].tolist(),
                                     marker_color=PALETTE[2]))
            fig_fs.add_trace(go.Bar(name="Riserva F-S", x=anni,
                                     y=res["riserve_per_anno"].tolist(),
                                     marker_color=PALETTE[0]))
            fig_fs.update_layout(barmode="stack",
                                  title="Pagato + Riserva F-S per Anno",
                                  xaxis_title="Anno Accadimento",
                                  yaxis_title="€", height=350, **PLOTLY_LAYOUT)
            st.plotly_chart(fig_fs, use_container_width=True)



# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 – CASE OUTSTANDING DEVELOPMENT
# ══════════════════════════════════════════════════════════════════════════════
with tab_co:
    st.subheader("🏛️ Case Outstanding Development")
    st.markdown("""
    Metodo di Wiser: analizza il rapporto tra **pagamenti incrementali** e
    **riserve di testa** aperte all'inizio del periodo. Adatto a linee claims-made
    o dove la quasi totalità dei sinistri è già nota (IBNR puro limitato).
    """)

    if "triangle" not in st.session_state:
        st.warning("⚠️ Prima inserisci e salva il triangolo dei pagamenti nella tab **Triangolo**.")
    else:
        tri_paid = st.session_state["triangle"]
        anni = st.session_state["anni_label"]
        n = st.session_state["n_anni"]

        st.markdown("**Triangolo delle riserve di testa — Case Outstanding (€)**")
        st.caption("Inserisci i valori di case outstanding per ogni anno di accadimento × sviluppo.")

        # Demo case outstanding coerente con demo pagamenti
        DEMO_CASE = {
            5: [
                [155_000, 35_000, 13_000, 4_000, 0],
                [145_000, 41_000, 11_000, 3_000, np.nan],
                [160_000, 38_000, 12_000, np.nan, np.nan],
                [165_000, 40_000, np.nan, np.nan, np.nan],
                [170_000, np.nan, np.nan, np.nan, np.nan],
            ],
        }
        default_case = DEMO_CASE.get(n, [[None] * n for _ in range(n)])

        case_data = render_triangle_input("case", n, anni, default_case,
                                           "Case Outstanding (€)", modalita)

        tipo_media_co = st.selectbox(
            "Tipo di media ratios",
            ["volume", "semplice", "mediana", "ultimi3"],
            format_func=lambda x: {"volume": "Media semplice", "semplice": "Semplice",
                                    "mediana": "Mediana", "ultimi3": "Ultimi 3 anni"}[x],
            key="tipo_media_co",
        )

        if st.button("▶️ Esegui Case Outstanding Development", type="primary"):
            case_array = build_triangle_from_session(case_data, n)
            try:
                res_co = case_outstanding_development(tri_paid, case_array, tipo_media_co)
                st.session_state["res_co"] = res_co
            except Exception as e:
                st.error(f"Errore: {e}")

        if "res_co" in st.session_state:
            res = st.session_state["res_co"]

            k1, k2 = st.columns(2)
            k1.metric("🟢 Riserva Totale CO", f"€ {res['riserva_totale']:,.0f}")
            k2.metric("Ratios selezionati", len(res["selected_ratios"]))

            st.divider()

            # Matrice ratios
            st.markdown("**Matrice ratios: Incrementale / Case Outstanding precedente**")
            ratio_df = pd.DataFrame(
                res["ratio_matrix"],
                index=anni,
                columns=[f"+{j}→+{j+1}" for j in range(n - 1)],
            )
            st.dataframe(ratio_df.style.format("{:.3f}", na_rep="—"), use_container_width=True)

            st.markdown("**Ratios selezionati per proiezione**")
            ratio_sel_df = pd.DataFrame({
                "Intervallo": [f"+{j}→+{j+1}" for j in range(n - 1)],
                "Ratio selezionato": [f"{r:.4f}" for r in res["selected_ratios"]],
            })
            st.dataframe(ratio_sel_df, use_container_width=True, hide_index=True)

            st.divider()
            df_co = pd.DataFrame({
                "Anno": anni,
                "Pagato attuale (€)": [f"€ {v:,.0f}" for v in res["pagati_attuali"]],
                "Case OS attuale (€)": [f"€ {v:,.0f}" for v in res["case_diagonale"]],
                "Ultimato CO (€)": [f"€ {v:,.0f}" for v in res["ultimati"]],
                "Riserva CO (€)": [f"€ {v:,.0f}" for v in res["riserve_per_anno"]],
            })
            st.dataframe(df_co, use_container_width=True, hide_index=True)

            fig = px.bar(x=anni, y=res["riserve_per_anno"].tolist(),
                         labels={"x": "Anno", "y": "Riserva CO (€)"},
                         title="Riserva IBNR — Case Outstanding Development",
                         color_discrete_sequence=[PALETTE[5]])
            fig.update_layout(height=320, **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 – EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.subheader("Δ Evaluation & Back-test Δ")

    if "triangle" not in st.session_state:
        st.warning("⚠️ Prima inserisci e salva il triangolo nella tab **Triangolo**.")
    else:
        tri = st.session_state["triangle"]
        anni = st.session_state["anni_label"]
        n = st.session_state["n_anni"]

        # ── Confronto metodi eseguiti ─────────────────────────────────────────
        st.markdown("### Confronto metodi eseguiti")
        METHOD_KEYS = [
            ("res_cl", "Chain Ladder"),
            ("res_bf", "Bornhuetter-Ferguson"),
            ("res_cc", "Cape Cod"),
            ("res_acpc", "Average Cost per Claim"),
            ("res_fs", "Frequency-Severity"),
            ("res_co", "Case Outstanding"),
        ]
        risultati = []
        for key, label in METHOD_KEYS:
            if key in st.session_state:
                risultati.append(st.session_state[key])

        if not risultati:
            st.info("Esegui almeno un metodo nelle tab precedenti.")
        else:
            def align(arr, target_n):
                a = list(arr)
                if len(a) >= target_n:
                    return a[:target_n]
                return a + [0.0] * (target_n - len(a))

            # Avvisa se anni non coincidono (ACPC con anni diversi)
            n_ref = n

            tabelle_data = {"Anno": anni}
            totali = {}
            for res in risultati:
                m = res["metodo"]
                ris = align(res["riserve_per_anno"], n_ref)
                tabelle_data[m] = [v for v in ris]
                totali[m] = sum(ris)

            df_comp = pd.DataFrame(tabelle_data)
            # Formatta
            df_show = df_comp.copy()
            for col in df_show.columns[1:]:
                df_show[col] = df_show[col].apply(lambda x: f"€ {x:,.0f}")
            # Riga totali
            totale_row = {"Anno": "TOTALE"}
            for m, t in totali.items():
                totale_row[m] = f"€ {t:,.0f}"
            df_show = pd.concat([df_show, pd.DataFrame([totale_row])], ignore_index=True)
            st.dataframe(df_show, use_container_width=True, hide_index=True)

            st.divider()

            # Grouped bar chart per anno
            fig_comp = go.Figure()
            for idx, res in enumerate(risultati):
                ris = align(res["riserve_per_anno"], n_ref)
                fig_comp.add_trace(go.Bar(
                    name=res["metodo"], x=anni, y=ris,
                    marker_color=PALETTE[idx % len(PALETTE)],
                ))
            fig_comp.update_layout(barmode="group",
                                    title="Confronto Riserve per Anno",
                                    xaxis_title="Anno Accadimento",
                                    yaxis_title="Riserva IBNR (€)",
                                    height=380, **PLOTLY_LAYOUT)
            st.plotly_chart(fig_comp, use_container_width=True)

            # Totali
            fig_tot = go.Figure(go.Bar(
                x=list(totali.keys()), y=list(totali.values()),
                marker_color=PALETTE[:len(totali)],
                text=[f"€ {v:,.0f}" for v in totali.values()],
                textposition="outside",
                textfont_color="#e0f2f1",
            ))
            fig_tot.update_layout(title="Riserva Totale per Metodo",
                                   yaxis_title="€", height=360, **PLOTLY_LAYOUT)
            st.plotly_chart(fig_tot, use_container_width=True)

            if len(totali) > 1:
                vals = list(totali.values())
                spread = max(vals) - min(vals)
                spread_pct = spread / max(vals) * 100
                colore = "warning-box" if spread_pct > 20 else "diagnostic-box"
                st.markdown(f"""<div class="{colore}">
                <strong>Range di stima:</strong> € {min(vals):,.0f} — € {max(vals):,.0f}
                &nbsp;|&nbsp; Spread: € {spread:,.0f} ({spread_pct:.1f}% del massimo)<br>
                {'⚠️ Spread elevato: le metodologie divergono significativamente. Analizzare le cause prima di selezionare la stima finale.' if spread_pct > 20 else '✅ Le stime sono ragionevolmente coerenti tra loro.'}
                </div>""", unsafe_allow_html=True)

        st.divider()

        # ── Back-test retrospettivo ───────────────────────────────────────────
        st.markdown("### 🕐 Back-test retrospettivo")
        st.markdown("""
        Rimuove le ultime *k* diagonali dal triangolo, ricalcola i metodi, e confronta
        gli ultimati proiettati con quelli stimati sul triangolo completo (proxy degli ultimati veri).
        """)

        col_bt1, col_bt2, col_bt3 = st.columns(3)
        n_diag = col_bt1.slider("Diagonali da rimuovere", 1, min(3, n - 2), 1, key="bt_diag")
        tipo_media_bt = col_bt2.selectbox(
            "Tipo media", ["volume", "semplice", "mediana", "ultimi3"],
            format_func=lambda x: {"volume": "Vol.-weighted", "semplice": "Semplice",
                                    "mediana": "Mediana", "ultimi3": "Ultimi 3"}[x],
            key="tipo_media_bt",
        )
        lr_bt = col_bt3.number_input("LR per BF/CC (%)", 50.0, 120.0, 70.0,
                                      step=1.0, key="lr_bt") / 100

        usa_premi_bt = st.checkbox("Usa premi BF/CC già inseriti (se disponibili)", value=True)

        premi_bt = None
        if usa_premi_bt:
            for pk in ["premi_bf", "premi_cc"]:
                if pk in st.session_state:
                    p = st.session_state[pk]
                    if len(p) == n:
                        premi_bt = np.array(p)
                        st.caption(f"Premi caricati da: {pk}")
                        break

        if st.button("▶️ Esegui Back-test", type="primary"):
            res_bt = backtest(tri, n_diag, premi_bt, lr_bt, tipo_media_bt)
            st.session_state["res_bt"] = res_bt

        if "res_bt" in st.session_state:
            res = st.session_state["res_bt"]

            st.markdown(f"**Diagonali rimosse: {res['n_diagonali_rimosse']}**")

            # Heatmap triangolo ridotto
            tri_rid_df = pd.DataFrame(
                res["triangolo_ridotto"],
                index=anni,
                columns=[f"Svil. +{j}" for j in range(n)],
            )
            fig_rid = go.Figure(go.Heatmap(
                z=tri_rid_df.values,
                x=tri_rid_df.columns.tolist(),
                y=tri_rid_df.index.tolist(),
                colorscale=[[0, "#071e2b"], [0.5, "#006064"], [1, "#80cbc4"]],
                text=np.where(np.isnan(tri_rid_df.values), "✂",
                              np.vectorize(lambda x: f"€{x:,.0f}")(
                                  np.where(np.isnan(tri_rid_df.values), 0, tri_rid_df.values))),
                texttemplate="%{text}",
            ))
            fig_rid.update_layout(title="Triangolo ridotto (celle ✂ = rimosse per il test)",
                                   height=300, **PLOTLY_LAYOUT)
            st.plotly_chart(fig_rid, use_container_width=True)

            # Risultati per metodo
            for metodo_nome, dati in res["metodi"].items():
                st.markdown(f"#### {metodo_nome}")
                df_bt = pd.DataFrame({
                    "Anno": anni,
                    "Ultimato proiettato (€)": [f"€ {v:,.0f}" for v in dati["ultimati_proiettati"]],
                    "Ultimato 'vero' (€)": [f"€ {v:,.0f}" for v in dati["ultimati_veri"]],
                    "Errore assoluto (€)": [f"€ {v:,.0f}" for v in dati["errori_assoluti"]],
                    "Errore %": [
                        f"{v:.1f}%" if not np.isnan(v) else "—"
                        for v in dati["errori_pct"]
                    ],
                })
                # Colora errori
                st.dataframe(df_bt, use_container_width=True, hide_index=True)

            # Chart comparativo errori percentuali
            fig_err = go.Figure()
            for metodo_nome, dati in res["metodi"].items():
                err = dati["errori_pct"]
                fig_err.add_trace(go.Bar(
                    name=metodo_nome, x=anni,
                    y=[v if not np.isnan(v) else 0 for v in err],
                    marker_color=PALETTE[list(res["metodi"].keys()).index(metodo_nome) % len(PALETTE)],
                ))
            fig_err.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
            fig_err.update_layout(
                barmode="group",
                title="Errore % per Anno e Metodo (positivo = sovrastima, negativo = sottostima)",
                xaxis_title="Anno Accadimento",
                yaxis_title="Errore %",
                height=370, **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_err, use_container_width=True)

            # RMSE per metodo
            st.markdown("**RMSE per metodo (proxy di accuratezza)**")
            rmse_data = {}
            for metodo_nome, dati in res["metodi"].items():
                err_abs = dati["errori_assoluti"]
                rmse = float(np.sqrt(np.nanmean(err_abs ** 2)))
                bias = float(np.nanmean(err_abs))
                rmse_data[metodo_nome] = {"RMSE (€)": f"€ {rmse:,.0f}", "Bias medio (€)": f"€ {bias:,.0f}"}

            df_rmse = pd.DataFrame(rmse_data).T.reset_index()
            df_rmse.columns = ["Metodo", "RMSE (€)", "Bias medio (€)"]
            st.dataframe(df_rmse, use_container_width=True, hide_index=True)

            st.markdown("""<div class="diagnostic-box">
            ℹ️ <strong>Nota metodologica</strong>: l'ultimato "vero" è qui approssimato con il 
            Chain Ladder sul triangolo completo. In un back-test reale, si userebbe 
            il dato osservato nelle diagonali successive. Usare questo test come indicatore 
            relativo di stabilità tra metodi, non come misura assoluta di accuratezza.
            </div>""", unsafe_allow_html=True)
