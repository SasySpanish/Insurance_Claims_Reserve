"""
diagnostics.py
──────────────
Diagnostica attuariale e report di evaluation per riserve sinistri.

Funzioni:
  - plot_development_triangle   : grafico sviluppo per accident year
  - plot_ldf_comparison         : confronto medie LDF
  - plot_paid_to_incurred       : P/I ratio per accident year
  - plot_closure_rates          : tasso di chiusura sinistri
  - detect_unstable_development : segnala anni/periodi anomali
  - generate_evaluation_report  : report HTML scaricabile

Dipendenze: plotly, pandas, numpy.
Non modifica nessuna funzione esistente.
"""

from __future__ import annotations

import base64
import datetime
import io
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from .riserva_sinistri import age_to_age_matrix
from .ldf_selection import select_ldf


# ──────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────────────────────

_PALETTE = [
    "#1f6bb0", "#e05c2a", "#2ca02c", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
]

_LAYOUT_BASE = dict(
    template="plotly_white",
    font=dict(family="Arial, sans-serif", size=12),
    margin=dict(l=60, r=30, t=50, b=50),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


def _fig_to_html(fig: go.Figure) -> str:
    """Converte una figura Plotly in stringa HTML embeddable (no <html> wrapper)."""
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def _fmt(x: float) -> str:
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.4f}"


# ──────────────────────────────────────────────────────────────────────────────
#  1. GRAFICO DI SVILUPPO (triangolo cumulato)
# ──────────────────────────────────────────────────────────────────────────────

def plot_development_triangle(
    triangle: np.ndarray,
    accident_years: Optional[list] = None,
    title: str = "Sviluppo Cumulato per Accident Year",
) -> tuple[go.Figure, pd.DataFrame]:
    """
    Curva di sviluppo cumulato per ogni accident year.

    Returns
    -------
    (fig, df_dati)  — figura Plotly + DataFrame dei dati utilizzati.
    """
    n = triangle.shape[0]
    if accident_years is None:
        accident_years = list(range(1, n + 1))

    records = []
    fig = go.Figure()

    for i, year in enumerate(accident_years):
        y_vals = [triangle[i, j] for j in range(n) if not np.isnan(triangle[i, j])]
        x_vals = list(range(1, len(y_vals) + 1))
        color = _PALETTE[i % len(_PALETTE)]

        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode="lines+markers",
            name=str(year),
            line=dict(color=color, width=2),
            marker=dict(size=6),
            hovertemplate=f"Anno {year}<br>Sviluppo: %{{x}}<br>Cumulato: %{{y:,.0f}}<extra></extra>",
        ))

        for j, (x, v) in enumerate(zip(x_vals, y_vals)):
            records.append({"Anno Accadimento": year, "Sviluppo": x, "Cumulato": v})

    fig.update_layout(
        **_LAYOUT_BASE,
        title=title,
        xaxis_title="Periodo di sviluppo",
        yaxis_title="Pagato cumulato (€)",
    )
    return fig, pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
#  2. CONFRONTO LDF (all-years vs weighted vs trimmed)
# ──────────────────────────────────────────────────────────────────────────────

def plot_ldf_comparison(
    riepilogo_medie: pd.DataFrame,
    title: str = "Confronto LDF — Medie a confronto",
) -> tuple[go.Figure, pd.DataFrame]:
    """
    Grafico a barre raggruppate per confrontare le diverse medie degli LDF.

    Parameters
    ----------
    riepilogo_medie : output di select_ldf()["riepilogo_medie"]

    Returns
    -------
    (fig, riepilogo_medie)
    """
    colonne_medie = ["All-Years", "Ultimi 3", "Ultimi 5", "Volume-Weighted", "Trimmed (10%)"]
    colonne_presenti = [c for c in colonne_medie if c in riepilogo_medie.columns]

    fig = go.Figure()
    for i, col in enumerate(colonne_presenti):
        fig.add_trace(go.Bar(
            name=col,
            x=riepilogo_medie.index.tolist(),
            y=riepilogo_medie[col].values,
            marker_color=_PALETTE[i % len(_PALETTE)],
            hovertemplate=f"{col}<br>Sviluppo: %{{x}}<br>LDF: %{{y:.4f}}<extra></extra>",
        ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=title,
        xaxis_title="Periodo di sviluppo",
        yaxis_title="Link Development Factor",
        barmode="group",
    )
    return fig, riepilogo_medie[colonne_presenti]


# ──────────────────────────────────────────────────────────────────────────────
#  3. PAID-TO-INCURRED RATIO
# ──────────────────────────────────────────────────────────────────────────────

def plot_paid_to_incurred(
    triangle_paid: np.ndarray,
    triangle_incurred: np.ndarray,
    accident_years: Optional[list] = None,
    title: str = "Paid-to-Incurred Ratio per Accident Year",
) -> tuple[go.Figure, pd.DataFrame]:
    """
    Rapporto Paid / Incurred per ogni accident year lungo lo sviluppo.
    Valori < 1 indicano riserve di testa significative; valori > 1 segnalano anomalie.

    Returns
    -------
    (fig, df_pi)
    """
    n = triangle_paid.shape[0]
    if accident_years is None:
        accident_years = list(range(1, n + 1))

    records = []
    fig = go.Figure()

    for i, year in enumerate(accident_years):
        x_vals, y_vals = [], []
        for j in range(n):
            p = triangle_paid[i, j]
            inc = triangle_incurred[i, j]
            if not np.isnan(p) and not np.isnan(inc) and inc > 0:
                ratio = p / inc
                x_vals.append(j + 1)
                y_vals.append(ratio)
                records.append({"Anno Accadimento": year, "Sviluppo": j + 1,
                                 "Paid": p, "Incurred": inc, "P/I": ratio})

        if x_vals:
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode="lines+markers",
                name=str(year),
                line=dict(color=_PALETTE[i % len(_PALETTE)], width=2),
                marker=dict(size=6),
                hovertemplate=f"Anno {year}<br>Sviluppo: %{{x}}<br>P/I: %{{y:.3f}}<extra></extra>",
            ))

    fig.add_hline(y=1.0, line_dash="dash", line_color="red",
                  annotation_text="P/I = 1", annotation_position="bottom right")
    fig.update_layout(
        **_LAYOUT_BASE,
        title=title,
        xaxis_title="Periodo di sviluppo",
        yaxis_title="Paid / Incurred",
    )
    return fig, pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
#  4. CLOSURE RATES
# ──────────────────────────────────────────────────────────────────────────────

def plot_closure_rates(
    triangle_closed: np.ndarray,
    triangle_reported: np.ndarray,
    accident_years: Optional[list] = None,
    title: str = "Closure Rate per Accident Year",
) -> tuple[go.Figure, pd.DataFrame]:
    """
    Tasso di chiusura: closed_claims / reported_claims per development period.

    Returns
    -------
    (fig, df_closure)
    """
    n = triangle_closed.shape[0]
    if accident_years is None:
        accident_years = list(range(1, n + 1))

    records = []
    fig = go.Figure()

    for i, year in enumerate(accident_years):
        x_vals, y_vals = [], []
        for j in range(n):
            cl = triangle_closed[i, j]
            rp = triangle_reported[i, j]
            if not np.isnan(cl) and not np.isnan(rp) and rp > 0:
                rate = cl / rp
                x_vals.append(j + 1)
                y_vals.append(rate)
                records.append({"Anno Accadimento": year, "Sviluppo": j + 1,
                                 "Chiusi": cl, "Denunciati": rp, "Closure Rate": rate})

        if x_vals:
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode="lines+markers",
                name=str(year),
                line=dict(color=_PALETTE[i % len(_PALETTE)], width=2),
                marker=dict(size=6),
                hovertemplate=f"Anno {year}<br>Sviluppo: %{{x}}<br>Closure: %{{y:.2%}}<extra></extra>",
            ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=title,
        xaxis_title="Periodo di sviluppo",
        yaxis_title="Closure Rate",
        yaxis=dict(tickformat=".0%"),
    )
    return fig, pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
#  5. RILEVAMENTO ANOMALIE DI SVILUPPO
# ──────────────────────────────────────────────────────────────────────────────

def detect_unstable_development(
    triangle: np.ndarray,
    accident_years: Optional[list] = None,
    cov_threshold: float = 0.15,
    reversal_threshold: float = 0.02,
) -> dict:
    """
    Segnala:
    - Periodi di sviluppo con alta variabilità inter-anno (CoV > soglia)
    - Accident year con inversioni significative nei LDF
      (LDF < 1 − reversal_threshold in almeno un periodo)

    Returns
    -------
    {
      "periodi_instabili": [(sviluppo, CoV), ...],
      "anni_anomali":      [(anno, sviluppo, ldf), ...],
      "riepilogo":         pd.DataFrame
    }
    """
    n = triangle.shape[0]
    if accident_years is None:
        accident_years = list(range(1, n + 1))

    ldf_matrix = age_to_age_matrix(triangle)
    periodi_instabili, anni_anomali = [], []
    records = []

    for col in range(ldf_matrix.shape[1]):
        vals = ldf_matrix[:, col]
        valid = vals[~np.isnan(vals)]
        if len(valid) < 2:
            continue
        mu = np.nanmean(valid)
        cov = np.nanstd(valid) / mu if mu > 0 else 0.0
        if cov > cov_threshold:
            periodi_instabili.append((f"{col+1}→{col+2}", round(float(cov), 4)))

        # Inversioni per accident year
        for i, year in enumerate(accident_years):
            v = ldf_matrix[i, col]
            if not np.isnan(v) and v < (1.0 - reversal_threshold):
                anni_anomali.append((year, f"{col+1}→{col+2}", round(float(v), 4)))

        records.append({
            "Sviluppo": f"{col+1}→{col+2}",
            "Media": round(float(mu), 4),
            "CoV": round(float(cov), 4),
            "Min": round(float(valid.min()), 4),
            "Max": round(float(valid.max()), 4),
            "Instabile": cov > cov_threshold,
        })

    return {
        "periodi_instabili": periodi_instabili,
        "anni_anomali": anni_anomali,
        "riepilogo": pd.DataFrame(records).set_index("Sviluppo"),
    }


# ──────────────────────────────────────────────────────────────────────────────
#  6. REPORT DI EVALUATION HTML
# ──────────────────────────────────────────────────────────────────────────────

def generate_evaluation_report(
    triangle_paid: np.ndarray,
    accident_years: Optional[list] = None,
    ldf_result: Optional[dict] = None,
    triangle_incurred: Optional[np.ndarray] = None,
    triangle_closed: Optional[np.ndarray] = None,
    triangle_reported: Optional[np.ndarray] = None,
    ramo: str = "",
    data_valutazione: Optional[str] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Genera un report HTML di evaluation con sezioni, grafici embedded e tabelle.

    Parameters
    ----------
    triangle_paid      : triangolo pagamenti cumulati (obbligatorio)
    accident_years     : lista anni accadimento
    ldf_result         : output di select_ldf() — se None viene calcolato internamente
    triangle_incurred  : triangolo sinistri ultimati (per P/I)
    triangle_closed    : triangolo sinistri chiusi (per closure rate)
    triangle_reported  : triangolo sinistri denunciati (per closure rate)
    ramo               : nome del ramo assicurativo (es. "RC Auto")
    data_valutazione   : stringa data (es. "31/12/2024"), default = oggi
    output_path        : se fornito, salva il file HTML su disco

    Returns
    -------
    Stringa HTML completa del report.
    """
    n = triangle_paid.shape[0]
    if accident_years is None:
        accident_years = list(range(1, n + 1))
    if data_valutazione is None:
        data_valutazione = datetime.date.today().strftime("%d/%m/%Y")

    # ── LDF ──────────────────────────────────────────────────────────────────
    if ldf_result is None:
        ldf_result = select_ldf(triangle_paid, accident_years=accident_years)

    ldf_sel    = ldf_result["ldf_selezionati"]
    cdf        = ldf_result["cdf_to_ultimate"]
    riepilogo  = ldf_result["riepilogo_medie"]
    giudizio   = ldf_result["giudizio"]
    outlier_info = ldf_result.get("outlier_info", {})

    # ── Anomalie ─────────────────────────────────────────────────────────────
    anomalie = detect_unstable_development(triangle_paid, accident_years)

    # ── Figure ───────────────────────────────────────────────────────────────
    fig_dev, _ = plot_development_triangle(triangle_paid, accident_years)
    fig_ldf, _ = plot_ldf_comparison(riepilogo)

    html_fig_dev = _fig_to_html(fig_dev)
    html_fig_ldf = _fig_to_html(fig_ldf)
    html_fig_pi = ""
    html_fig_cl = ""

    if triangle_incurred is not None:
        fig_pi, _ = plot_paid_to_incurred(triangle_paid, triangle_incurred, accident_years)
        html_fig_pi = _fig_to_html(fig_pi)

    if triangle_closed is not None and triangle_reported is not None:
        fig_cl, _ = plot_closure_rates(triangle_closed, triangle_reported, accident_years)
        html_fig_cl = _fig_to_html(fig_cl)

    # ── Tabella LDF selezionati ───────────────────────────────────────────────
    df_ldf_tab = riepilogo.copy()
    df_ldf_tab["LDF Selezionato"] = ldf_sel
    df_ldf_tab["CDF to Ult."]     = cdf[:-1]  # esclude ultimo (= 1.0)

    # ── CSS + template HTML ───────────────────────────────────────────────────
    css = """
    <style>
      body { font-family: Arial, sans-serif; font-size: 13px; color: #1a1a1a;
             max-width: 1200px; margin: 0 auto; padding: 30px; }
      h1   { color: #1f3c6b; font-size: 22px; border-bottom: 2px solid #1f3c6b;
             padding-bottom: 6px; }
      h2   { color: #1f6bb0; font-size: 16px; margin-top: 32px;
             border-left: 4px solid #1f6bb0; padding-left: 10px; }
      h3   { font-size: 14px; color: #444; margin-top: 20px; }
      table { border-collapse: collapse; width: 100%; margin-top: 12px; }
      th   { background-color: #1f3c6b; color: #fff; padding: 7px 10px;
             text-align: center; font-size: 12px; }
      td   { border: 1px solid #ddd; padding: 5px 10px; text-align: right;
             font-size: 12px; }
      tr:nth-child(even) td { background-color: #f5f8ff; }
      td.label { text-align: left; font-weight: bold; }
      .tag-ok   { background: #d4edda; color: #155724; border-radius: 4px;
                  padding: 2px 6px; font-size: 11px; }
      .tag-warn { background: #fff3cd; color: #856404; border-radius: 4px;
                  padding: 2px 6px; font-size: 11px; }
      .tag-err  { background: #f8d7da; color: #721c24; border-radius: 4px;
                  padding: 2px 6px; font-size: 11px; }
      .meta     { color: #666; font-size: 12px; margin-top: 4px; }
      .note     { background: #eef4ff; border-left: 3px solid #1f6bb0;
                  padding: 8px 12px; margin-top: 12px; font-size: 12px; }
      .section  { margin-bottom: 40px; }
      hr        { border: none; border-top: 1px solid #ccc; margin: 30px 0; }
    </style>
    """

    def _df_to_html_table(df: pd.DataFrame, fmt: dict | None = None) -> str:
        styled = df.copy()
        if fmt:
            for col, f in fmt.items():
                if col in styled.columns:
                    styled[col] = styled[col].map(f)
        return styled.to_html(classes="", border=0, escape=False)

    # ── Sezione giudizio ─────────────────────────────────────────────────────
    critiche_html = "".join(
        f'<li><span class="tag-warn">⚠</span> {c}</li>'
        for c in giudizio["criticita"]
    )
    outlier_html = "".join(
        f'<li>Anno {o["anno_accadimento"]}, sviluppo {o["sviluppo"]}: LDF = {o["valore"]:.4f}</li>'
        for o in giudizio.get("outlier_dettaglio", [])
    ) or "<li>Nessun outlier rilevato.</li>"

    anomalie_periodi_html = "".join(
        f'<li>Sviluppo {p}: CoV = {c:.2%}</li>'
        for p, c in anomalie["periodi_instabili"]
    ) or "<li>Nessun periodo instabile rilevato.</li>"

    anomalie_anni_html = "".join(
        f'<li>Anno {y}, sviluppo {d}: LDF = {v:.4f} (inversione)</li>'
        for y, d, v in anomalie["anni_anomali"]
    ) or "<li>Nessun anno anomalo rilevato.</li>"

    # ── Tabella riepilogo diagnostica sviluppo ────────────────────────────────
    diag_df = anomalie["riepilogo"].copy()
    diag_df["Instabile"] = diag_df["Instabile"].map(
        lambda x: '<span class="tag-warn">Sì</span>' if x
                  else '<span class="tag-ok">No</span>'
    )
    diag_table_html = diag_df.to_html(border=0, escape=False)

    # ── LDF table ─────────────────────────────────────────────────────────────
    fmt_4 = lambda x: f"{x:.4f}" if pd.notna(x) else ""
    fmt_pct = lambda x: f"{x:.1%}" if pd.notna(x) else ""
    ldf_tab = df_ldf_tab.copy()
    for c in ["All-Years", "Ultimi 3", "Ultimi 5", "Volume-Weighted",
              "Trimmed (10%)", "LDF Selezionato", "CDF to Ult."]:
        if c in ldf_tab.columns:
            ldf_tab[c] = ldf_tab[c].map(fmt_4)
    if "CoV" in ldf_tab.columns:
        ldf_tab["CoV"] = ldf_tab["CoV"].map(fmt_pct)
    ldf_table_html = ldf_tab.to_html(border=0, escape=False)

    optional_sections = ""
    if html_fig_pi:
        optional_sections += f"""
        <div class="section">
          <h2>4. Paid-to-Incurred Ratio</h2>
          {html_fig_pi}
        </div>
        """
    if html_fig_cl:
        optional_sections += f"""
        <div class="section">
          <h2>5. Closure Rates</h2>
          {html_fig_cl}
        </div>
        """

    # ── Assemblaggio HTML ─────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="UTF-8">
  <title>Report di Evaluation — {ramo or 'Riserva Sinistri'}</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  {css}
</head>
<body>
  <h1>Report di Evaluation — Riserva Sinistri</h1>
  <p class="meta">
    <strong>Ramo:</strong> {ramo or 'N/D'} &nbsp;|&nbsp;
    <strong>Data di valutazione:</strong> {data_valutazione} &nbsp;|&nbsp;
    <strong>Anni accadimento:</strong> {accident_years[0]}–{accident_years[-1]}
  </p>
  <hr>

  <!-- SEZIONE 1: LDF -->
  <div class="section">
    <h2>1. Selezione degli LDF</h2>
    <h3>1.1 Riepilogo medie e LDF selezionati</h3>
    {ldf_table_html}
    <h3>1.2 Giudizio attuariale sulla selezione</h3>
    <table>
      <tr><th class="label" style="width:200px">Parametro</th><th>Valore</th></tr>
      <tr><td class="label">Metodo scelto</td><td>{giudizio["metodo_scelto"]}</td></tr>
      <tr><td class="label">Anni esclusi</td>
          <td>{", ".join(str(y) for y in giudizio["anni_esclusi"]) or "Nessuno"}</td></tr>
      <tr><td class="label">Tail factor</td><td>{giudizio["tail_factor"]:.4f}</td></tr>
      <tr><td class="label">Outlier rimossi</td>
          <td>{"Sì" if giudizio["outlier_rimossi"] else "No"}
              ({giudizio["n_outlier"]} rilevati)</td></tr>
    </table>
    <h3>1.3 Criticità</h3>
    <ul>{critiche_html}</ul>
    <h3>1.4 Outlier rilevati</h3>
    <ul>{outlier_html}</ul>
  </div>
  <hr>

  <!-- SEZIONE 2: Grafici sviluppo e LDF -->
  <div class="section">
    <h2>2. Grafici di sviluppo</h2>
    <h3>2.1 Triangolo cumulato per Accident Year</h3>
    {html_fig_dev}
    <h3>2.2 Confronto medie LDF</h3>
    {html_fig_ldf}
  </div>
  <hr>

  <!-- SEZIONE 3: Diagnostica anomalie -->
  <div class="section">
    <h2>3. Diagnostica attuariale</h2>
    <h3>3.1 Stabilità dello sviluppo per periodo</h3>
    {diag_table_html}
    <h3>3.2 Periodi instabili (CoV &gt; {15:.0f}%)</h3>
    <ul>{anomalie_periodi_html}</ul>
    <h3>3.3 Anni con inversioni LDF</h3>
    <ul>{anomalie_anni_html}</ul>
  </div>
  <hr>

  {optional_sections}

  <!-- FOOTER -->
  <p class="meta" style="margin-top:40px; color:#999; font-size:11px;">
    Report generato il {datetime.date.today().strftime("%d/%m/%Y")} —
    modulo <em>diagnostics.py</em> / riserva sinistri.
  </p>
</body>
</html>
"""

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

    return html


# ══════════════════════════════════════════════════════════════════════════════
#  ESEMPIO MINIMO DI UTILIZZO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    T = np.array([
        [1_000, 1_500, 1_750, 1_850, 1_900, 1_920],
        [1_100, 1_650, 1_900, 2_000, 2_050,  np.nan],
        [1_200, 1_780, 2_050, 2_150,  np.nan,  np.nan],
        [1_300, 1_900, 2_200,  np.nan,  np.nan,  np.nan],
        [1_400, 2_050,  np.nan,  np.nan,  np.nan,  np.nan],
        [1_500,  np.nan,  np.nan,  np.nan,  np.nan,  np.nan],
    ], dtype=float)

    anni = list(range(2018, 2024))

    # Diagnostica standalone
    anomalie = detect_unstable_development(T, anni)
    print("Periodi instabili:", anomalie["periodi_instabili"])
    print("Anni anomali:", anomalie["anni_anomali"])

    # Report HTML
    html = generate_evaluation_report(
        triangle_paid=T,
        accident_years=anni,
        ramo="RC Auto",
        data_valutazione="31/12/2024",
        output_path="evaluation_report.html",
    )
    print(f"Report generato: {len(html):,} caratteri.")
