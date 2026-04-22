"""Calcolo Riserva Sinistri
Metodi: Chain Ladder, Bornhuetter-Ferguson, Cape Cod, Average Cost per Claim
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import (
    chain_ladder, bornhuetter_ferguson, cape_cod,
    average_cost_per_claim, tabella_riepilogo_riserve,
)

st.set_page_config(page_title="Riserva Sinistri", page_icon="📐", layout="wide")

st.title("📐 Calcolo Riserva Sinistri")
st.caption("Stima IBNR con Chain Ladder, Bornhuetter-Ferguson, Cape Cod e Average Cost per Claim.")

# ── HELPER per costruire triangolo da input ────────────────────────────────────
def build_triangle_from_session(data: list, n: int) -> np.ndarray:
    tri = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i + j < n:
                tri[i, j] = data[i][j]
    return tri


# ── TAB principali ─────────────────────────────────────────────────────────────
tab_input, tab_cl, tab_bf, tab_cc, tab_acpc, tab_compare = st.tabs([
    "📥 Triangolo Input",
    "⛓️ Chain Ladder",
    "📊 Bornhuetter-Ferguson",
    "🎣 Cape Cod",
    "🔢 Avg Cost/Claim",
    "🔍 Confronto Metodi",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – TRIANGOLO INPUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_input:
    st.subheader("Triangolo dei Pagamenti Cumulati")
    st.markdown("""
    Inserisci i pagamenti **cumulati** per anno di accadimento × anno di sviluppo.
    Le celle grigie (diagonale superiore destra) sono dati futuri — non compilarle.
    """)


    c1, c2 = st.columns([1, 3])
    with c1:
        n_anni = st.slider("N° anni (dimensione triangolo)", 3, 8, 5)
        anno_base = st.number_input("Primo anno di accadimento", 2016, 2023, 2019)
        anni_label = [str(int(anno_base) + i) for i in range(n_anni)]

        st.divider()
        st.markdown("**Modalità inserimento**")
        modalita = st.radio("", ["Manuale", "Demo precaricato"], label_visibility="collapsed")

    with c2:
        # Demo data (triangolo tipico RC Auto)
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


      

    
        if modalita == "Demo precaricato" and n_anni in DEMO:
            default_data = DEMO[n_anni]
            
        elif modalita == "Demo precaricato":
            # Genera demo casuale
            rng = np.random.default_rng(0)
            base = rng.integers(150_000, 300_000, n_anni)
            default_data = []
            for i in range(n_anni):
                row = [None] * n_anni
                for j in range(n_anni - i):
                    factor = 1 + 0.55 * (j == 0) + 0.12 * (j == 1) + 0.05 * (j >= 2)
                    row[j] = float(base[i]) * (1.0 + sum(
                        [0.55 if k == 0 else 0.12 if k == 1 else 0.04 for k in range(j)]
                    ))
                default_data.append(row)
        else:
            default_data = [[None] * n_anni for _ in range(n_anni)]
        
        st.markdown("**Inserisci il triangolo cumulato (€)**")
        header_cols = st.columns([1] + [1] * n_anni)
        header_cols[0].markdown("**Anno Acc. \\ Svil.**")
        for j in range(n_anni):
            header_cols[j + 1].markdown(f"**Anno +{j}**")

        triangle_data = []
        for i in range(n_anni):
            row_cols = st.columns([1] + [1] * n_anni)
            row_cols[0].markdown(f"**{anni_label[i]}**")
            row = []
            for j in range(n_anni):
                is_future = (i + j >= n_anni)
                if is_future:
                    row_cols[j + 1].markdown("▫️")
                    row.append(np.nan)
                else:
                    default_val = default_data[i][j] if default_data[i][j] is not None else 0.0
                    val = row_cols[j + 1].number_input(
                        "", min_value=0.0,
                        value=float(default_val) if not (isinstance(default_val, float) and np.isnan(default_val)) else 0.0,
                        step=1000.0, format="%.0f",
                        key=f"tri_{modalita}_{i}_{j}",
                        label_visibility="collapsed",
                        disabled=is_future,
                    )
                    row.append(val)
            triangle_data.append(row)

        if st.button("💾 Salva triangolo", type="primary"):
            tri_array = build_triangle_from_session(triangle_data, n_anni)
            st.session_state["triangle"] = tri_array
            st.session_state["n_anni"] = n_anni
            st.session_state["anni_label"] = anni_label
            st.success("Triangolo salvato! Vai alle tab dei metodi.")

    # Visualizza triangolo come heatmap
    if "triangle" in st.session_state:
        st.divider()
        st.subheader("🔥 Heatmap Triangolo")
        tri_df = pd.DataFrame(
            st.session_state["triangle"],
            index=st.session_state["anni_label"],
            columns=[f"Svil. +{j}" for j in range(st.session_state["n_anni"])]
        )
        fig_heat = go.Figure(go.Heatmap(
            z=tri_df.values,
            x=tri_df.columns.tolist(),
            y=tri_df.index.tolist(),
            colorscale="Blues",
            text=np.where(np.isnan(tri_df.values), "", tri_df.values.astype(int).astype(str)),
            texttemplate="%{text}",
            hovertemplate="Anno %{y} | %{x}: € %{z:,.0f}<extra></extra>",
        ))
        fig_heat.update_layout(
            title="Triangolo Cumulato dei Pagamenti",
            height=350,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_heat, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – CHAIN LADDER
# ══════════════════════════════════════════════════════════════════════════════
with tab_cl:
    st.subheader("⛓️ Chain Ladder")
    st.markdown("""
    Il metodo Chain Ladder proietta i pagamenti futuri moltiplicando i valori della
    diagonale attuale per i **fattori di sviluppo** età-età calcolati storicamente.
    """)

    if "triangle" not in st.session_state:
        st.warning("⚠️ Prima inserisci e salva il triangolo nella tab **Triangolo Input**.")
    else:
        tri = st.session_state["triangle"]
        anni = st.session_state["anni_label"]
        n = st.session_state["n_anni"]

        if st.button("▶️ Esegui Chain Ladder", type="primary"):
            res_cl = chain_ladder(tri)
            st.session_state["res_cl"] = res_cl

        if "res_cl" in st.session_state:
            res = st.session_state["res_cl"]

            # KPI
            k1, k2 = st.columns(2)
            k1.metric("🟢 Riserva Totale IBNR", f"€ {res['riserva_totale']:,.0f}")
            k2.metric("Fattori di sviluppo calcolati", len(res["fattori_sviluppo"]))

            st.divider()

            # Tabella fattori
            col_f, col_r = st.columns(2)
            with col_f:
                st.markdown("**Fattori età-età (Age-to-Age)**")
                df_fact = pd.DataFrame({
                    "Sviluppo": [f"+{j} → +{j+1}" for j in range(len(res["fattori_sviluppo"]))],
                    "Fattore": [f"{f:.4f}" for f in res["fattori_sviluppo"]],
                })
                st.dataframe(df_fact, use_container_width=True, hide_index=True)

            with col_r:
                st.markdown("**Riserve per anno di accadimento**")
                df_riserve = pd.DataFrame({
                    "Anno Accadimento": anni,
                    "Pagato Attuale (€)": [f"€ {v:,.0f}" for v in res["pagati_attuali"]],
                    "Ultimato CL (€)": [f"€ {v:,.0f}" for v in res["ultimati"]],
                    "Riserva IBNR (€)": [f"€ {v:,.0f}" for v in res["riserve_per_anno"]],
                })
                st.dataframe(df_riserve, use_container_width=True, hide_index=True)

            # Triangolo completato – heatmap
            st.divider()
            st.markdown("**Triangolo Completato (proiezione CL)**")
            tri_comp = res["triangolo_completato"]
            labels_is_projected = [
                [True if i + j >= n else False for j in range(n)]
                for i in range(n)
            ]
            fig_tri = go.Figure()
            # Valori osservati
            z_obs = np.where(
                np.array(labels_is_projected), np.nan, tri_comp
            )
            z_proj = np.where(
                np.array(labels_is_projected), tri_comp, np.nan
            )
            fig_tri.add_trace(go.Heatmap(
                z=z_obs, colorscale="Blues",
                x=[f"+{j}" for j in range(n)], y=anni,
                name="Osservato", showscale=False,
                hovertemplate="Osservato | %{y} +%{x}: € %{z:,.0f}<extra></extra>",
            ))
            fig_tri.add_trace(go.Heatmap(
                z=z_proj, colorscale="Oranges",
                x=[f"+{j}" for j in range(n)], y=anni,
                name="Proiettato", showscale=True,
                hovertemplate="Proiettato | %{y} +%{x}: € %{z:,.0f}<extra></extra>",
            ))
            fig_tri.update_layout(
                title="Triangolo completato — Blu=osservato, Arancio=proiettato",
                height=380, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_tri, use_container_width=True)

            # Bar chart riserve per anno
            fig_bar = px.bar(
                x=anni, y=res["riserve_per_anno"].tolist(),
                labels={"x": "Anno Accadimento", "y": "Riserva IBNR (€)"},
                title="Riserva IBNR per Anno (Chain Ladder)",
                color_discrete_sequence=["#636efa"],
            )
            fig_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_bar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – BORNHUETTER-FERGUSON
# ══════════════════════════════════════════════════════════════════════════════
with tab_bf:
    st.subheader("📊 Bornhuetter-Ferguson")
    st.markdown("""
    Il metodo BF combina l'esperienza storica (Chain Ladder) con un **loss ratio a priori**
    applicato ai premi di rischio. Riduce la volatilità degli anni più recenti.
    """)

    if "triangle" not in st.session_state:
        st.warning("⚠️ Prima inserisci e salva il triangolo nella tab **Triangolo Input**.")
    else:
        tri = st.session_state["triangle"]
        anni = st.session_state["anni_label"]
        n = st.session_state["n_anni"]

        col_p, col_lr = st.columns(2)
        with col_p:
            st.markdown("**Premi di rischio per anno (€)**")
            premi_bf = []
            for i, anno in enumerate(anni):
                p = st.number_input(f"Premio {anno}", min_value=0.0,
                                    value=float(300_000 + i * 10_000),
                                    step=10_000.0, key=f"prem_bf_{i}")
                premi_bf.append(p)
        with col_lr:
            lr_atteso = st.number_input("Loss Ratio a priori (%)", 0.0, 200.0, 65.0, step=1.0)
            st.caption("Es.: 65% significa che ci si aspetta che il 65% dei premi venga pagato in sinistri.")

        if st.button("▶️ Esegui Bornhuetter-Ferguson", type="primary"):
            res_bf = bornhuetter_ferguson(tri, np.array(premi_bf), lr_atteso / 100)
            st.session_state["res_bf"] = res_bf

        if "res_bf" in st.session_state:
            res = st.session_state["res_bf"]

            k1, k2, k3 = st.columns(3)
            k1.metric("🟢 Riserva Totale BF", f"€ {res['riserva_totale']:,.0f}")
            k2.metric("Loss Ratio a priori", f"{res['loss_ratio_atteso']*100:.1f}%")
            k3.metric("Perdita a priori totale", f"€ {res['perdita_apriori'].sum():,.0f}")

            st.divider()
            df_bf = pd.DataFrame({
                "Anno": anni,
                "CDF (tail-to-ult)": [f"{v:.3f}" for v in res["cdfs"]],
                "% Sviluppato": [f"{v*100:.1f}%" for v in res["pct_sviluppato"]],
                "Perdita a priori (€)": [f"€ {v:,.0f}" for v in res["perdita_apriori"]],
                "Pagato Attuale (€)": [f"€ {v:,.0f}" for v in res["pagati_attuali"]],
                "Ultimato BF (€)": [f"€ {v:,.0f}" for v in res["ultimati"]],
                "Riserva BF (€)": [f"€ {v:,.0f}" for v in res["riserve_per_anno"]],
            })
            st.dataframe(df_bf, use_container_width=True, hide_index=True)

            fig = px.bar(x=anni, y=res["riserve_per_anno"].tolist(),
                         labels={"x": "Anno", "y": "Riserva BF (€)"},
                         title="Riserva IBNR — Bornhuetter-Ferguson",
                         color_discrete_sequence=["#ab63fa"])
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – CAPE COD
# ══════════════════════════════════════════════════════════════════════════════
with tab_cc:
    st.subheader("🎣 Cape Cod")
    st.markdown("""
    Il metodo Cape Cod stima il **loss ratio emergente** direttamente dai dati osservati
    (ELR = Expected Loss Ratio), eliminando la soggettività nell'ipotesi a priori.
    """)

    if "triangle" not in st.session_state:
        st.warning("⚠️ Prima inserisci e salva il triangolo nella tab **Triangolo Input**.")
    else:
        tri = st.session_state["triangle"]
        anni = st.session_state["anni_label"]
        n = st.session_state["n_anni"]

        st.markdown("**Premi di rischio per anno (€)**")
        cols_cc = st.columns(n)
        premi_cc = []
        for i, anno in enumerate(anni):
            p = cols_cc[i].number_input(anno, min_value=0.0, value=float(300_000 + i * 10_000),
                                         step=10_000.0, key=f"prem_cc_{i}")
            premi_cc.append(p)

        if st.button("▶️ Esegui Cape Cod", type="primary"):
            res_cc = cape_cod(tri, np.array(premi_cc))
            st.session_state["res_cc"] = res_cc

        if "res_cc" in st.session_state:
            res = st.session_state["res_cc"]

            k1, k2 = st.columns(2)
            k1.metric("🟢 Riserva Totale CC", f"€ {res['riserva_totale']:,.0f}")
            k2.metric("ELR Stimato", f"{res['elr_stimato']*100:.2f}%")

            st.divider()
            df_cc = pd.DataFrame({
                "Anno": anni,
                "% Sviluppato": [f"{v*100:.1f}%" for v in res["pct_sviluppato"]],
                "Perdita a priori CC (€)": [f"€ {v:,.0f}" for v in res["perdita_apriori"]],
                "Pagato Attuale (€)": [f"€ {v:,.0f}" for v in res["pagati_attuali"]],
                "Ultimato CC (€)": [f"€ {v:,.0f}" for v in res["ultimati"]],
                "Riserva CC (€)": [f"€ {v:,.0f}" for v in res["riserve_per_anno"]],
            })
            st.dataframe(df_cc, use_container_width=True, hide_index=True)

            fig = px.bar(x=anni, y=res["riserve_per_anno"].tolist(),
                         labels={"x": "Anno", "y": "Riserva CC (€)"},
                         title="Riserva IBNR — Cape Cod",
                         color_discrete_sequence=["#ffa15a"])
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 – AVERAGE COST PER CLAIM
# ══════════════════════════════════════════════════════════════════════════════
with tab_acpc:
    st.subheader("🔢 Average Cost per Claim (ACPC)")
    st.markdown("""
    Stima la riserva come: **N° sinistri IBNR × Costo Medio** (con eventuale inflazione).
    Non richiede il triangolo: utile per rami con frequenza stabile e costo medio misurabile.
    """)

    n_acpc = st.slider("N° anni di accadimento", 3, 8, 5, key="n_acpc")
    anno_base_acpc = st.number_input("Primo anno", 2016, 2023, 2019, key="ab_acpc")
    anni_acpc = [int(anno_base_acpc) + i for i in range(n_acpc)]
    anno_val = st.number_input("Anno di valutazione", 2022, 2030,
                                int(anno_base_acpc) + n_acpc - 1 + 1)

    col_a, col_b, col_c = st.columns(3)
    costo_medio = col_a.number_input("Costo medio per sinistro (€)", 100.0, 500_000.0,
                                      8_000.0, step=500.0)
    inflazione = col_b.number_input("Inflazione annua (%)", 0.0, 20.0, 3.0, step=0.5)

    st.markdown("**Numero sinistri IBNR stimati per anno**")
    cols_acpc = st.columns(n_acpc)
    n_sin_acpc = []
    for i in range(n_acpc):
        val = cols_acpc[i].number_input(str(anni_acpc[i]), min_value=0.0,
                                         value=float(50 - i * 5),
                                         step=1.0, key=f"nacpc_{i}")
        n_sin_acpc.append(val)

    if st.button("▶️ Esegui ACPC", type="primary"):
        res_acpc = average_cost_per_claim(
            np.array(n_sin_acpc), costo_medio, inflazione,
            anni_acpc, int(anno_val)
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
            "Sin. IBNR": res["numero_sinistri_ibnr"],
            "Costo Inf. (€)": [f"€ {v:,.0f}" for v in res["costi_inflazionati"]],
            "Riserva (€)": [f"€ {v:,.0f}" for v in res["riserve_per_anno"]],
        })
        st.dataframe(df_acpc, use_container_width=True, hide_index=True)

        fig = px.bar(
            x=[str(a) for a in anni_acpc], y=res["riserve_per_anno"].tolist(),
            labels={"x": "Anno", "y": "Riserva (€)"},
            title="Riserva IBNR — Average Cost per Claim",
            color_discrete_sequence=["#19d3f3"],
        )
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 – CONFRONTO METODI
# ══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.subheader("🔍 Confronto tra Metodi")

    risultati = []
    for key, label in [("res_cl", "Chain Ladder"), ("res_bf", "Bornhuetter-Ferguson"),
                        ("res_cc", "Cape Cod"), ("res_acpc", "Average Cost per Claim")]:
        if key in st.session_state:
            risultati.append(st.session_state[key])

    if not risultati:
        st.info("Esegui almeno un metodo nelle tab precedenti per vedere il confronto.")
    else:
        # Anni comuni
        anni_ref = st.session_state.get("anni_label", [str(i) for i in range(5)])
        n_ref = len(anni_ref)

        # Allineamento lunghezze
        def align(arr, n):
            a = list(arr)
            if len(a) >= n:
                return a[:n]
            return a + [0.0] * (n - len(a))

        # Tabella confronto
        tabelle_data = {"Anno": anni_ref}
        totali = {}
        for res in risultati:
            m = res["metodo"]
            ris = align(res["riserve_per_anno"], n_ref)
            tabelle_data[f"{m} (€)"] = [f"€ {v:,.0f}" for v in ris]
            totali[m] = sum(ris)

        df_comp = pd.DataFrame(tabelle_data)
        # Riga totali
        totale_row = {"Anno": "**TOTALE**"}
        for m, t in totali.items():
            totale_row[f"{m} (€)"] = f"€ {t:,.0f}"
        df_comp = pd.concat([df_comp, pd.DataFrame([totale_row])], ignore_index=True)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        st.divider()

        # Grouped bar chart per anno
        fig_comp = go.Figure()
        colors = ["#636efa", "#ab63fa", "#ffa15a", "#19d3f3"]
        for idx, res in enumerate(risultati):
            ris = align(res["riserve_per_anno"], n_ref)
            fig_comp.add_trace(go.Bar(
                name=res["metodo"],
                x=anni_ref,
                y=ris,
                marker_color=colors[idx % len(colors)],
            ))
        fig_comp.update_layout(
            barmode="group",
            title="Confronto Riserve per Anno di Accadimento",
            xaxis_title="Anno Accadimento",
            yaxis_title="Riserva IBNR (€)",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            height=420,
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # Totali a confronto
        fig_tot = go.Figure(go.Bar(
            x=list(totali.keys()),
            y=list(totali.values()),
            marker_color=colors[:len(totali)],
            text=[f"€ {v:,.0f}" for v in totali.values()],
            textposition="outside",
        ))
        fig_tot.update_layout(
            title="Riserva Totale per Metodo",
            yaxis_title="€",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            height=380,
        )
        st.plotly_chart(fig_tot, use_container_width=True)

        # Range di incertezza
        if len(totali) > 1:
            vals = list(totali.values())
            st.info(
                f"📏 **Range di stima:** "
                f"€ {min(vals):,.0f} — € {max(vals):,.0f} "
                f"(scarto: € {max(vals)-min(vals):,.0f}, "
                f"{(max(vals)-min(vals))/max(vals)*100:.1f}%)"
            )
