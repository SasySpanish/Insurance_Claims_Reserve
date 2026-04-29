"""
utils/diagnostica.py
Modulo di diagnostica attuariale per riserve sinistri.

Funzioni principali:
  - select_ldf            : selezione LDF con giudizio attuariale
  - plot_development      : grafici di sviluppo per accident year
  - plot_ldf_comparison   : confronto medie LDF
  - plot_paid_to_incurred : ratio paid/incurred nel tempo
  - plot_closure_rates    : tasso di chiusura sinistri
  - detect_anomalies      : segnalazione anomalie di sviluppo
  - generate_evaluation_report : report HTML/PDF scaricabile
"""

from __future__ import annotations

import base64
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


# ── Palette coerente con l'app ─────────────────────────────────────────────────
_PALETTE = ["#80cbc4", "#4db6ac", "#26a69a", "#00897b", "#00695c",
            "#b2dfdb", "#e0f2f1", "#26c6da", "#0097a7", "#006064"]
_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(255,255,255,1)",  # bianco per report
    font_color="#1a1a2e",
    font_family="DM Sans, sans-serif",
    xaxis=dict(gridcolor="rgba(0,0,0,0.07)", zerolinecolor="rgba(0,0,0,0.1)"),
    yaxis=dict(gridcolor="rgba(0,0,0,0.07)", zerolinecolor="rgba(0,0,0,0.1)"),
    legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="rgba(0,0,0,0.1)",
                borderwidth=1),
)


# ══════════════════════════════════════════════════════════════════════════════
#  DATACLASS: risultato select_ldf
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LDFSelection:
    """Risultato strutturato della selezione LDF."""
    selected: np.ndarray                        # LDF selezionati (n-1,)
    summary: pd.DataFrame                       # tutte le medie per colonna
    outliers: dict[int, list[int]]              # {col: [row_indices outlier]}
    excluded_years: list[int]                   # indici anni esclusi
    method: str                                 # metodo di selezione usato
    outlier_method: str                         # IQR o zscore
    high_cv_cols: list[int]                     # colonne ad alta variabilità (CV > 0.1)
    notes: list[str] = field(default_factory=list)  # note diagnostiche testuali


# ══════════════════════════════════════════════════════════════════════════════
#  1. SELEZIONE LDF
# ══════════════════════════════════════════════════════════════════════════════

def _link_ratio_matrix(triangle: np.ndarray) -> np.ndarray:
    """
    Restituisce la matrice (n × n-1) dei link ratio f_{i,j} = C_{i,j+1} / C_{i,j}.
    NaN dove non disponibile.
    """
    n = triangle.shape[0]
    mat = np.full((n, n - 1), np.nan)
    for j in range(n - 1):
        valid_rows = n - 1 - j
        with np.errstate(invalid="ignore", divide="ignore"):
            ratios = triangle[:valid_rows, j + 1] / triangle[:valid_rows, j]
        ratios[triangle[:valid_rows, j] == 0] = np.nan
        mat[:valid_rows, j] = ratios
    return mat


def _mean_all(ratios: np.ndarray) -> float:
    return float(np.nanmean(ratios))


def _mean_last_k(ratios: np.ndarray, k: int) -> float:
    valid = ratios[~np.isnan(ratios)]
    return float(np.mean(valid[-k:])) if len(valid) >= 1 else np.nan


def _mean_weighted(triangle: np.ndarray, col: int) -> float:
    n = triangle.shape[0]
    valid_rows = n - 1 - col
    num = np.nansum(triangle[:valid_rows, col + 1])
    den = np.nansum(triangle[:valid_rows, col])
    return float(num / den) if den > 0 else np.nan


def _detect_outliers_iqr(ratios: np.ndarray) -> np.ndarray:
    """Restituisce maschera booleana True = outlier (metodo IQR)."""
    valid = ratios[~np.isnan(ratios)]
    if len(valid) < 4:
        return np.zeros(len(ratios), dtype=bool)
    q1, q3 = np.nanpercentile(ratios, 25), np.nanpercentile(ratios, 75)
    iqr = q3 - q1
    return (ratios < q1 - 1.5 * iqr) | (ratios > q3 + 1.5 * iqr)


def _detect_outliers_zscore(ratios: np.ndarray, threshold: float = 2.0) -> np.ndarray:
    """Restituisce maschera booleana True = outlier (metodo Z-score)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu, sigma = np.nanmean(ratios), np.nanstd(ratios)
    if sigma == 0:
        return np.zeros(len(ratios), dtype=bool)
    return np.abs((ratios - mu) / sigma) > threshold


def select_ldf(
    triangle: np.ndarray,
    anni_label: list[str],
    method: str = "weighted",
    outlier_method: str = "iqr",
    remove_outliers: bool = True,
    exclude_years: Optional[list[int]] = None,
    tail_factor: float = 1.0,
) -> LDFSelection:
    """
    Selezione LDF con giudizio attuariale.

    Parameters
    ----------
    triangle        : triangolo cumulato n×n.
    anni_label      : etichette anni di accadimento.
    method          : 'weighted' | 'all' | 'last3' | 'last5' | 'trimmed'
                      'trimmed' = media all-years escludendo outlier.
    outlier_method  : 'iqr' | 'zscore'
    remove_outliers : se True, esclude outlier dal metodo selezionato.
    exclude_years   : lista di indici (0-based) di anni da escludere manualmente.
    tail_factor     : fattore di coda da applicare all'ultimo CDF.

    Returns
    -------
    LDFSelection
    """
    n = triangle.shape[0]
    exclude_years = exclude_years or []
    lrm = _link_ratio_matrix(triangle)

    # Maschera righe escluse
    excl_mask = np.zeros(n, dtype=bool)
    excl_mask[exclude_years] = True

    outlier_fn = _detect_outliers_iqr if outlier_method == "iqr" else _detect_outliers_zscore

    summary_rows = []
    selected = []
    outliers_dict: dict[int, list[int]] = {}
    high_cv_cols: list[int] = []
    notes: list[str] = []

    for col in range(n - 1):
        col_ratios = lrm[:, col].copy()

        # Escludi anni manuali
        col_ratios[excl_mask] = np.nan

        # Rileva outlier
        out_mask = outlier_fn(col_ratios)
        out_indices = list(np.where(out_mask & ~np.isnan(col_ratios))[0])
        if out_indices:
            outliers_dict[col] = out_indices
            out_labels = [anni_label[i] for i in out_indices]
            notes.append(f"Colonna +{col}→+{col+1}: outlier rilevati in {out_labels} "
                         f"({outlier_method.upper()})")

        # Versione senza outlier per metodi che li escludono
        col_clean = col_ratios.copy()
        if remove_outliers:
            col_clean[out_mask] = np.nan

        # Tutte le medie
        m_all      = _mean_all(col_ratios)
        m_all_cl   = _mean_all(col_clean)
        m_last3    = _mean_last_k(col_clean, 3)
        m_last5    = _mean_last_k(col_clean, 5)
        m_weighted = _mean_weighted(triangle, col)  # volume-weighted sempre sull'intero

        # Coefficiente di variazione (variabilità)
        cv = float(np.nanstd(col_clean) / np.nanmean(col_clean)) if np.nanmean(col_clean) > 0 else 0.0
        if cv > 0.10:
            high_cv_cols.append(col)
            notes.append(f"Colonna +{col}→+{col+1}: alta variabilità (CV={cv:.2%}), "
                         "selezionare con cautela.")

        # Selezione in base al metodo
        method_map = {
            "all":      m_all_cl,
            "last3":    m_last3,
            "last5":    m_last5,
            "weighted": m_weighted,
            "trimmed":  m_all_cl,
        }
        sel = method_map.get(method, m_weighted)
        # Fallback se NaN (es. last5 su triangolo piccolo)
        if np.isnan(sel):
            sel = m_weighted
            notes.append(f"Colonna +{col}→+{col+1}: fallback su weighted (dati insufficienti "
                         f"per '{method}').")

        selected.append(sel)
        summary_rows.append({
            "Intervallo":        f"+{col}→+{col+1}",
            "All-years":         round(m_all, 5),
            "All-years (pulito)":round(m_all_cl, 5),
            "Ultimi 3":          round(m_last3, 5) if not np.isnan(m_last3) else np.nan,
            "Ultimi 5":          round(m_last5, 5) if not np.isnan(m_last5) else np.nan,
            "Volume-weighted":   round(m_weighted, 5),
            "Selezionato":       round(sel, 5),
            "CV":                f"{cv:.2%}",
            "Outlier rilevati":  len(out_indices),
        })

    selected_arr = np.array(selected) * tail_factor  # tail applicato all'ultimo step
    # Correzione: tail va solo sull'ultimo fattore selezionato
    selected_arr = np.array(selected)
    if len(selected_arr) > 0:
        selected_arr[-1] = selected_arr[-1] * tail_factor

    summary_df = pd.DataFrame(summary_rows)

    if exclude_years:
        ex_labels = [anni_label[i] for i in exclude_years if i < len(anni_label)]
        notes.append(f"Anni esclusi manualmente: {ex_labels}.")

    return LDFSelection(
        selected=selected_arr,
        summary=summary_df,
        outliers=outliers_dict,
        excluded_years=exclude_years,
        method=method,
        outlier_method=outlier_method,
        high_cv_cols=high_cv_cols,
        notes=notes,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  2. DIAGNOSTICA — GRAFICI
# ══════════════════════════════════════════════════════════════════════════════

def plot_development(
    triangle: np.ndarray,
    anni_label: list[str],
) -> tuple[go.Figure, pd.DataFrame]:
    """
    Grafici di sviluppo cumulato per accident year.
    Restituisce (figura, DataFrame dei dati usati).
    """
    n = triangle.shape[0]
    dev_ages = [f"+{j}" for j in range(n)]

    fig = go.Figure()
    data_records = []

    for i, anno in enumerate(anni_label):
        y_vals = [triangle[i, j] if j <= n - 1 - i else None for j in range(n)]
        x_vals = dev_ages[:n - i]
        y_plot = [v for v in y_vals if v is not None]

        fig.add_trace(go.Scatter(
            x=x_vals, y=y_plot,
            mode="lines+markers",
            name=anno,
            line=dict(color=_PALETTE[i % len(_PALETTE)], width=2),
            marker=dict(size=6),
            hovertemplate=f"Anno {anno} | %{{x}}: € %{{y:,.0f}}<extra></extra>",
        ))

        for j, v in enumerate(y_plot):
            data_records.append({"Anno": anno, "Sviluppo": dev_ages[j], "Cumulato": v})

    fig.update_layout(
        title="Sviluppo cumulato per Accident Year",
        xaxis_title="Età di sviluppo",
        yaxis_title="Pagamenti cumulati (€)",
        height=420,
        **{k: v for k, v in _LAYOUT.items()},
    )
    return fig, pd.DataFrame(data_records)


def plot_ldf_comparison(
    triangle: np.ndarray,
    anni_label: list[str],
    ldf_selection: Optional[LDFSelection] = None,
) -> tuple[go.Figure, pd.DataFrame]:
    """
    Confronto grafico tra medie LDF: all-years, weighted, last3, last5,
    e LDF selezionati se forniti.
    """
    n = triangle.shape[0]
    lrm = _link_ratio_matrix(triangle)
    dev_labels = [f"+{j}→+{j+1}" for j in range(n - 1)]

    all_y, weighted_y, last3_y, last5_y = [], [], [], []
    for col in range(n - 1):
        r = lrm[:, col]
        all_y.append(_mean_all(r))
        weighted_y.append(_mean_weighted(triangle, col))
        last3_y.append(_mean_last_k(r, 3))
        last5_y.append(_mean_last_k(r, 5))

    fig = go.Figure()
    for name, y, color in [
        ("All-years",       all_y,      _PALETTE[0]),
        ("Volume-weighted", weighted_y, _PALETTE[1]),
        ("Ultimi 3",        last3_y,    _PALETTE[3]),
        ("Ultimi 5",        last5_y,    _PALETTE[4]),
    ]:
        fig.add_trace(go.Scatter(
            x=dev_labels, y=y, mode="lines+markers", name=name,
            line=dict(color=color, width=2), marker=dict(size=7),
        ))

    if ldf_selection is not None:
        fig.add_trace(go.Scatter(
            x=dev_labels, y=ldf_selection.selected.tolist(),
            mode="lines+markers", name="Selezionati",
            line=dict(color="#ff6b6b", width=3, dash="dash"),
            marker=dict(size=9, symbol="diamond"),
        ))

    fig.update_layout(
        title="Confronto medie LDF per intervallo di sviluppo",
        xaxis_title="Intervallo",
        yaxis_title="LDF",
        height=380,
        **{k: v for k, v in _LAYOUT.items()},
    )

    df_out = pd.DataFrame({
        "Intervallo": dev_labels,
        "All-years": all_y,
        "Volume-weighted": weighted_y,
        "Ultimi 3": last3_y,
        "Ultimi 5": last5_y,
    })
    if ldf_selection is not None:
        df_out["Selezionati"] = ldf_selection.selected
    return fig, df_out


def plot_paid_to_incurred(
    triangle_paid: np.ndarray,
    triangle_incurred: np.ndarray,
    anni_label: list[str],
) -> tuple[go.Figure, pd.DataFrame]:
    """
    Paid-to-Incurred ratio per accident year nel tempo.
    ratio_{i,j} = paid_{i,j} / incurred_{i,j}
    """
    n = triangle_paid.shape[0]
    dev_ages = [f"+{j}" for j in range(n)]

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio_tri = np.where(
            triangle_incurred > 0,
            triangle_paid / triangle_incurred,
            np.nan,
        )

    fig = go.Figure()
    records = []

    for i, anno in enumerate(anni_label):
        y = [ratio_tri[i, j] if j <= n - 1 - i and not np.isnan(ratio_tri[i, j])
             else None for j in range(n)]
        y_plot = [v for v in y if v is not None]
        x_plot = dev_ages[:len(y_plot)]

        fig.add_trace(go.Scatter(
            x=x_plot, y=y_plot,
            mode="lines+markers", name=anno,
            line=dict(color=_PALETTE[i % len(_PALETTE)], width=2),
            marker=dict(size=6),
            hovertemplate=f"Anno {anno} | %{{x}}: %{{y:.2%}}<extra></extra>",
        ))
        for j, v in enumerate(y_plot):
            records.append({"Anno": anno, "Sviluppo": x_plot[j], "Paid/Incurred": v})

    fig.add_hline(y=1.0, line_dash="dash", line_color="rgba(255,100,100,0.5)",
                  annotation_text="Paid = Incurred")
    fig.update_layout(
        title="Paid-to-Incurred Ratio per Accident Year",
        xaxis_title="Età di sviluppo",
        yaxis_title="Paid / Incurred",
        yaxis_tickformat=".0%",
        height=380,
        **{k: v for k, v in _LAYOUT.items()},
    )
    return fig, pd.DataFrame(records)


def plot_closure_rates(
    triangle_closed: np.ndarray,
    triangle_reported: np.ndarray,
    anni_label: list[str],
) -> tuple[go.Figure, pd.DataFrame]:
    """
    Tasso di chiusura sinistri: closed / reported per accident year nel tempo.
    """
    n = triangle_closed.shape[0]
    dev_ages = [f"+{j}" for j in range(n)]

    with np.errstate(invalid="ignore", divide="ignore"):
        closure_tri = np.where(
            triangle_reported > 0,
            triangle_closed / triangle_reported,
            np.nan,
        )

    fig = go.Figure()
    records = []

    for i, anno in enumerate(anni_label):
        y_plot = [closure_tri[i, j] for j in range(n - i)
                  if not np.isnan(closure_tri[i, j])]
        x_plot = dev_ages[:len(y_plot)]

        fig.add_trace(go.Scatter(
            x=x_plot, y=y_plot,
            mode="lines+markers", name=anno,
            line=dict(color=_PALETTE[i % len(_PALETTE)], width=2),
            marker=dict(size=6),
            hovertemplate=f"Anno {anno} | %{{x}}: %{{y:.1%}}<extra></extra>",
        ))
        for j, v in enumerate(y_plot):
            records.append({"Anno": anno, "Sviluppo": x_plot[j], "Closure Rate": v})

    fig.update_layout(
        title="Closure Rate (Sinistri Chiusi / Denunciati) per Accident Year",
        xaxis_title="Età di sviluppo",
        yaxis_title="Closure Rate",
        yaxis_tickformat=".0%",
        height=380,
        **{k: v for k, v in _LAYOUT.items()},
    )
    return fig, pd.DataFrame(records)


def detect_anomalies(
    triangle: np.ndarray,
    anni_label: list[str],
    outlier_method: str = "iqr",
    cv_threshold: float = 0.10,
) -> dict:
    """
    Identifica anomalie di sviluppo nel triangolo.

    Returns
    -------
    dict con chiavi:
      - 'outlier_cells'   : lista di (anno, intervallo, valore) per LDF outlier
      - 'unstable_cols'   : intervalli con CV > cv_threshold
      - 'anomalous_years' : anni con almeno 2 LDF outlier
      - 'summary'         : DataFrame riepilogativo
    """
    n = triangle.shape[0]
    lrm = _link_ratio_matrix(triangle)
    outlier_fn = _detect_outliers_iqr if outlier_method == "iqr" else _detect_outliers_zscore

    outlier_cells = []
    unstable_cols = []
    year_outlier_count = np.zeros(n, dtype=int)

    for col in range(n - 1):
        col_r = lrm[:, col]
        out_mask = outlier_fn(col_r)
        cv = float(np.nanstd(col_r) / np.nanmean(col_r)) if np.nanmean(col_r) > 0 else 0.0

        if cv > cv_threshold:
            unstable_cols.append({
                "Intervallo": f"+{col}→+{col+1}",
                "CV": f"{cv:.2%}",
            })

        for row in np.where(out_mask & ~np.isnan(col_r))[0]:
            outlier_cells.append({
                "Anno":       anni_label[row],
                "Intervallo": f"+{col}→+{col+1}",
                "LDF":        round(float(lrm[row, col]), 5),
            })
            year_outlier_count[row] += 1

    anomalous_years = [
        anni_label[i] for i in range(n) if year_outlier_count[i] >= 2
    ]

    summary = pd.DataFrame({
        "Anno": anni_label,
        "N° LDF outlier": year_outlier_count,
        "Anomalo": ["⚠️ Sì" if c >= 2 else "✅ No" for c in year_outlier_count],
    })

    return {
        "outlier_cells":   outlier_cells,
        "unstable_cols":   unstable_cols,
        "anomalous_years": anomalous_years,
        "summary":         summary,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  3. REPORT DI EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def _fig_to_b64(fig: go.Figure, width: int = 900, height: int = 380) -> str:
    """Converte una figura Plotly in stringa base64 PNG per embedding HTML."""
    img_bytes = pio.to_image(fig, format="png", width=width, height=height, scale=2)
    return base64.b64encode(img_bytes).decode("utf-8")


def _section(title: str, content: str) -> str:
    return f"""
    <section>
      <h2>{title}</h2>
      {content}
    </section>
    """


def _table_html(df: pd.DataFrame) -> str:
    return df.to_html(index=False, border=0, classes="report-table",
                      na_rep="—", float_format=lambda x: f"{x:,.5f}")


def generate_evaluation_report(
    triangle: np.ndarray,
    anni_label: list[str],
    ldf_selection: LDFSelection,
    riserve_risultati: Optional[list[dict]] = None,
    anomalie: Optional[dict] = None,
    triangle_incurred: Optional[np.ndarray] = None,
    triangle_closed: Optional[np.ndarray] = None,
    triangle_reported: Optional[np.ndarray] = None,
    titolo: str = "Report di Valutazione Riserve Sinistri",
    produrre_pdf: bool = False,
) -> tuple[bytes, Optional[bytes]]:
    """
    Genera un report di valutazione attuariale in HTML (e opzionalmente PDF).

    Parameters
    ----------
    triangle            : triangolo pagamenti cumulati.
    anni_label          : etichette anni di accadimento.
    ldf_selection       : output di select_ldf().
    riserve_risultati   : lista di dict output dei metodi (chain_ladder, bf, ecc.).
    anomalie            : output di detect_anomalies().
    triangle_incurred   : per paid-to-incurred (opzionale).
    triangle_closed     : per closure rates (opzionale).
    triangle_reported   : per closure rates (opzionale).
    titolo              : titolo del report.
    produrre_pdf        : se True, tenta di convertire in PDF con weasyprint.

    Returns
    -------
    (html_bytes, pdf_bytes) — pdf_bytes è None se produrre_pdf=False o fallisce.
    """
    from datetime import date

    sections_html = []

    # ── 1. Header ─────────────────────────────────────────────────────────────
    sections_html.append(f"""
    <div class="report-header">
      <h1>{titolo}</h1>
      <p class="subtitle">Data di valutazione: {date.today().strftime("%d %m %Y")} &nbsp;|&nbsp;
         Dimensione triangolo: {triangle.shape[0]}×{triangle.shape[0]}</p>
    </div>
    """)

    # ── 2. LDF selezionati ────────────────────────────────────────────────────
    sections_html.append(_section(
        "1. LDF Selezionati",
        f"""
        <p><strong>Metodo:</strong> {ldf_selection.method} &nbsp;|&nbsp;
           <strong>Outlier:</strong> {ldf_selection.outlier_method.upper()} &nbsp;|&nbsp;
           <strong>Anni esclusi:</strong>
           {[anni_label[i] for i in ldf_selection.excluded_years] or 'nessuno'}
        </p>
        {_table_html(ldf_selection.summary)}
        """
    ))

    # ── 3. Note diagnostiche selezione ────────────────────────────────────────
    if ldf_selection.notes:
        notes_html = "<ul>" + "".join(f"<li>{n}</li>" for n in ldf_selection.notes) + "</ul>"
        sections_html.append(_section("2. Note sulla Selezione LDF", notes_html))

    # ── 4. Grafico sviluppo ───────────────────────────────────────────────────
    fig_dev, _ = plot_development(triangle, anni_label)
    img_dev = _fig_to_b64(fig_dev)
    sections_html.append(_section(
        "3. Sviluppo Cumulato per Accident Year",
        f'<img src="data:image/png;base64,{img_dev}" style="width:100%;max-width:900px"/>'
    ))

    # ── 5. Confronto LDF ──────────────────────────────────────────────────────
    fig_ldf, df_ldf = plot_ldf_comparison(triangle, anni_label, ldf_selection)
    img_ldf = _fig_to_b64(fig_ldf)
    sections_html.append(_section(
        "4. Confronto Medie LDF",
        f'<img src="data:image/png;base64,{img_ldf}" style="width:100%;max-width:900px"/>'
        + _table_html(df_ldf.round(5))
    ))

    # ── 6. Paid-to-Incurred (opzionale) ───────────────────────────────────────
    if triangle_incurred is not None:
        fig_pi, df_pi = plot_paid_to_incurred(triangle, triangle_incurred, anni_label)
        img_pi = _fig_to_b64(fig_pi)
        sections_html.append(_section(
            "5. Paid-to-Incurred Ratio",
            f'<img src="data:image/png;base64,{img_pi}" style="width:100%;max-width:900px"/>'
        ))

    # ── 7. Closure rates (opzionale) ──────────────────────────────────────────
    if triangle_closed is not None and triangle_reported is not None:
        fig_cr, _ = plot_closure_rates(triangle_closed, triangle_reported, anni_label)
        img_cr = _fig_to_b64(fig_cr)
        sections_html.append(_section(
            "6. Closure Rate",
            f'<img src="data:image/png;base64,{img_cr}" style="width:100%;max-width:900px"/>'
        ))

    # ── 8. Anomalie ───────────────────────────────────────────────────────────
    if anomalie:
        anom_content = ""
        if anomalie["anomalous_years"]:
            anom_content += (f"<p>⚠️ <strong>Anni anomali (≥2 LDF outlier):</strong> "
                             f"{anomalie['anomalous_years']}</p>")
        if anomalie["unstable_cols"]:
            anom_content += "<p><strong>Intervalli instabili:</strong></p>"
            anom_content += _table_html(pd.DataFrame(anomalie["unstable_cols"]))
        if anomalie["outlier_cells"]:
            anom_content += "<p><strong>LDF outlier individuali:</strong></p>"
            anom_content += _table_html(pd.DataFrame(anomalie["outlier_cells"]))
        if not anom_content:
            anom_content = "<p>✅ Nessuna anomalia rilevata.</p>"
        sections_html.append(_section("7. Anomalie Rilevate", anom_content))

    # ── 9. Sintesi riserve (opzionale) ────────────────────────────────────────
    if riserve_risultati:
        rows = []
        for res in riserve_risultati:
            rows.append({
                "Metodo": res.get("metodo", "—"),
                "Riserva Totale (€)": f"€ {res.get('riserva_totale', 0):,.0f}",
            })
        sections_html.append(_section(
            "8. Sintesi Riserve per Metodo",
            _table_html(pd.DataFrame(rows))
        ))

    # ── CSS + assemblaggio ────────────────────────────────────────────────────
    css = """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;600&family=DM+Mono&display=swap');
      body { font-family: 'DM Sans', sans-serif; color: #1a1a2e; margin: 0; padding: 0;
             background: #f8f9fa; }
      .report-header { background: linear-gradient(135deg, #0a2a3a, #00695c);
                       color: white; padding: 40px 60px; }
      .report-header h1 { font-size: 2rem; margin: 0 0 8px 0; font-weight: 600; }
      .subtitle { color: rgba(255,255,255,0.75); margin: 0; font-size: 0.9rem; }
      section { padding: 32px 60px; border-bottom: 1px solid #e0e0e0; }
      section:nth-child(even) { background: #ffffff; }
      h2 { color: #00695c; font-size: 1.2rem; font-weight: 600;
           border-left: 4px solid #00897b; padding-left: 12px; }
      p { line-height: 1.6; color: #333; }
      .report-table { border-collapse: collapse; width: 100%; font-size: 0.85rem;
                      font-family: 'DM Mono', monospace; margin-top: 12px; }
      .report-table th { background: #00695c; color: white; padding: 8px 12px;
                         text-align: left; font-weight: 500; }
      .report-table td { padding: 6px 12px; border-bottom: 1px solid #e0e0e0; }
      .report-table tr:hover td { background: #e8f5e9; }
      img { border-radius: 8px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); margin: 12px 0; }
      ul { color: #444; line-height: 1.8; }
      li { margin-bottom: 4px; }
    </style>
    """

    html = f"""<!DOCTYPE html>
    <html lang="it">
    <head>
      <meta charset="UTF-8"/>
      <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
      <title>{titolo}</title>
      {css}
    </head>
    <body>
      {''.join(sections_html)}
    </body>
    </html>"""

    html_bytes = html.encode("utf-8")

    # ── PDF opzionale ─────────────────────────────────────────────────────────
    pdf_bytes: Optional[bytes] = None
    if produrre_pdf:
        try:
            from weasyprint import HTML as WeasyprintHTML
            pdf_bytes = WeasyprintHTML(string=html).write_pdf()
        except ImportError:
            warnings.warn(
                "weasyprint non installato. Aggiungi 'weasyprint' a requirements.txt. "
                "Il PDF non è stato generato.",
                stacklevel=2,
            )
        except Exception as exc:
            warnings.warn(f"Generazione PDF fallita: {exc}", stacklevel=2)

    return html_bytes, pdf_bytes


def generate_evaluation_report_full(
    triangle: np.ndarray,
    anni_label: list[str],
    riserve_risultati: list[dict],
    backtest_result: Optional[dict] = None,
    ldf_selection: Optional[LDFSelection] = None,
    titolo: str = "Report di Valutazione Riserve — Evaluation",
    produrre_pdf: bool = False,
) -> tuple[bytes, Optional[bytes]]:
    """
    Report di Evaluation scaricabile: confronto metodi, grafici, back-test.
    Separato dal report diagnostica. Stile consulenziale scuro/teal.
    """
    from datetime import date

    # ── CSS ───────────────────────────────────────────────────────────────────
    css = """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

      *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

      body {
        font-family: 'DM Sans', sans-serif;
        background: #0d3d4a;
        color: #e0f2f1;
        font-size: 14px;
        line-height: 1.6;
      }

      /* ── Header ── */
      .report-header {
        background: linear-gradient(135deg, #071e2b 0%, #0a2f3a 60%, #0d3d4a 100%);
        padding: 48px 64px 40px;
        border-bottom: 2px solid #00897b;
        position: relative;
      }
      .report-header::after {
        content: '';
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #00897b, #4db6ac, #80cbc4, #4db6ac, #00897b);
      }
      .report-header h1 {
        font-size: 2rem;
        font-weight: 600;
        color: #b2dfdb;
        letter-spacing: -0.5px;
        margin-bottom: 8px;
      }
      .report-header .meta {
        color: rgba(178,223,219,0.6);
        font-size: 0.82rem;
        display: flex;
        gap: 24px;
        flex-wrap: wrap;
      }
      .report-header .meta span {
        display: flex;
        align-items: center;
        gap: 6px;
      }
      .badge {
        display: inline-block;
        background: rgba(0,137,123,0.25);
        border: 1px solid rgba(128,203,196,0.3);
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.75rem;
        color: #80cbc4;
        margin-left: 8px;
      }

      /* ── Navigazione sezioni ── */
      .toc {
        background: #0a2a3a;
        border-left: 3px solid #00897b;
        padding: 20px 32px;
        margin: 0;
      }
      .toc h3 { color: #80cbc4; font-size: 0.78rem; text-transform: uppercase;
                letter-spacing: 1px; margin-bottom: 10px; }
      .toc ol { padding-left: 18px; color: #4db6ac; font-size: 0.85rem; }
      .toc li { margin-bottom: 4px; }

      /* ── Sezione ── */
      section {
        padding: 40px 64px;
        border-bottom: 1px solid rgba(128,203,196,0.1);
      }
      section:nth-child(even) {
        background: rgba(10,42,58,0.4);
      }

      h2 {
        font-size: 1.1rem;
        font-weight: 600;
        color: #80cbc4;
        letter-spacing: 0.3px;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(128,203,196,0.2);
        display: flex;
        align-items: center;
        gap: 10px;
      }
      h2 .section-num {
        background: linear-gradient(135deg, #00897b, #00695c);
        color: white;
        border-radius: 50%;
        width: 28px; height: 28px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.78rem;
        font-weight: 600;
        flex-shrink: 0;
      }

      h3 {
        font-size: 0.9rem;
        font-weight: 500;
        color: #4db6ac;
        margin: 20px 0 10px;
      }

      p { color: #b2dfdb; margin-bottom: 10px; }

      /* ── KPI cards ── */
      .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
      }
      .kpi-card {
        background: rgba(0,137,123,0.12);
        border: 1px solid rgba(128,203,196,0.2);
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
      }
      .kpi-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #80cbc4;
        margin-bottom: 6px;
      }
      .kpi-value {
        font-family: 'DM Mono', monospace;
        font-size: 1.2rem;
        font-weight: 500;
        color: #e0f2f1;
      }
      .kpi-delta {
        font-size: 0.75rem;
        color: #4db6ac;
        margin-top: 4px;
      }

      /* ── Tabelle ── */
      .report-table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'DM Mono', monospace;
        font-size: 0.8rem;
        margin: 16px 0;
        border-radius: 10px;
        overflow: hidden;
      }
      .report-table thead tr {
        background: linear-gradient(135deg, #00695c, #00897b);
      }
      .report-table th {
        padding: 10px 14px;
        text-align: left;
        color: #e0f2f1;
        font-weight: 500;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.78rem;
        letter-spacing: 0.3px;
      }
      .report-table td {
        padding: 9px 14px;
        border-bottom: 1px solid rgba(128,203,196,0.1);
        color: #b2dfdb;
      }
      .report-table tr:nth-child(even) td {
        background: rgba(0,137,123,0.06);
      }
      .report-table tr:last-child td {
        background: rgba(0,137,123,0.18);
        color: #e0f2f1;
        font-weight: 600;
        border-bottom: none;
      }

      /* ── Alert box ── */
      .alert {
        border-radius: 10px;
        padding: 14px 18px;
        margin: 16px 0;
        font-size: 0.85rem;
        display: flex;
        align-items: flex-start;
        gap: 10px;
      }
      .alert-warning {
        background: rgba(255,160,0,0.1);
        border: 1px solid rgba(255,160,0,0.3);
        color: #ffcc80;
      }
      .alert-ok {
        background: rgba(0,137,123,0.12);
        border: 1px solid rgba(128,203,196,0.25);
        color: #80cbc4;
      }
      .alert-icon { font-size: 1rem; flex-shrink: 0; margin-top: 1px; }

      /* ── Grafici ── */
      .chart-container {
        background: rgba(7,30,43,0.5);
        border: 1px solid rgba(128,203,196,0.15);
        border-radius: 12px;
        padding: 4px;
        margin: 16px 0;
        overflow: hidden;
      }
      .chart-container img {
        width: 100%;
        border-radius: 10px;
        display: block;
      }
      .chart-caption {
        font-size: 0.75rem;
        color: rgba(128,203,196,0.6);
        text-align: center;
        padding: 6px 0 2px;
      }

      /* ── Note LDF ── */
      .note-list { list-style: none; padding: 0; }
      .note-list li {
        padding: 8px 12px;
        border-left: 3px solid #00897b;
        background: rgba(0,137,123,0.08);
        margin-bottom: 6px;
        border-radius: 0 6px 6px 0;
        color: #b2dfdb;
        font-size: 0.83rem;
      }

      /* ── Footer ── */
      .report-footer {
        background: #071e2b;
        padding: 24px 64px;
        text-align: center;
        color: rgba(128,203,196,0.4);
        font-size: 0.75rem;
        border-top: 1px solid rgba(128,203,196,0.1);
      }

      /* ── Spread indicator ── */
      .spread-bar {
        height: 6px;
        border-radius: 3px;
        background: rgba(128,203,196,0.15);
        margin: 8px 0;
        overflow: hidden;
      }
      .spread-fill {
        height: 100%;
        border-radius: 3px;
        background: linear-gradient(90deg, #00897b, #4db6ac);
      }
    </style>
    """

    # ── Palette grafici (sfondo scuro per report) ──────────────────────────────
    DARK_LAYOUT = dict(
        plot_bgcolor="#0a2a3a",
        paper_bgcolor="#0a2a3a",
        font_color="#e0f2f1",
        font_family="DM Sans",
        xaxis=dict(gridcolor="rgba(128,203,196,0.1)", zerolinecolor="rgba(128,203,196,0.15)"),
        yaxis=dict(gridcolor="rgba(128,203,196,0.1)", zerolinecolor="rgba(128,203,196,0.15)"),
        legend=dict(bgcolor="rgba(7,30,43,0.8)", bordercolor="rgba(128,203,196,0.2)", borderwidth=1),
    )
    PAL = ["#80cbc4", "#4db6ac", "#26a69a", "#00897b", "#00695c",
           "#b2dfdb", "#26c6da", "#0097a7", "#006064", "#e0f2f1"]

    sections = []

    # ── Header ────────────────────────────────────────────────────────────────
    n_metodi = len(riserve_risultati)
    totali = [r.get("riserva_totale", 0) for r in riserve_risultati]
    riserva_min = min(totali) if totali else 0
    riserva_max = max(totali) if totali else 0
    spread_pct = (riserva_max - riserva_min) / riserva_max * 100 if riserva_max > 0 else 0

    sections.append(f"""
    <div class="report-header">
      <h1>{titolo}</h1>
      <div class="meta">
        <span>• Data: {date.today().strftime("%d/%m/%Y")}</span>
        <span>• Triangolo: {triangle.shape[0]}×{triangle.shape[0]}</span>
        <span>• Metodi eseguiti: {n_metodi}</span>
        <span>• Anni: {anni_label[0]}–{anni_label[-1]}</span>
      </div>
    </div>
    """)

    # ── Indice ────────────────────────────────────────────────────────────────
    toc_items = [
        "Sintesi Riserve per Metodo",
        "Confronto Riserve per Anno",
        "Analisi dello Spread",
        "LDF Selezionati" if ldf_selection else None,
        "Back-test Retrospettivo" if backtest_result else None,
    ]
    toc_html = "".join(
        f"<li>{item}</li>" for item in toc_items if item
    )
    sections.append(f"""
    <div class="toc">
      <h3>Contenuto del Report</h3>
      <ol>{toc_html}</ol>
    </div>
    """)

    # ── 1. Sintesi KPI + tabella ───────────────────────────────────────────────
    kpi_cards = "".join(f"""
      <div class="kpi-card">
        <div class="kpi-label">{r['metodo']}</div>
        <div class="kpi-value">€ {r['riserva_totale']:,.0f}</div>
      </div>
    """ for r in riserve_risultati)

    n_anni = triangle.shape[0]
    table_rows = ""
    for i, anno in enumerate(anni_label):
        cells = ""
        for r in riserve_risultati:
            rpp = r.get("riserve_per_anno", [])
            v = rpp[i] if i < len(rpp) else 0.0
            cells += f"<td>€ {v:,.0f}</td>"
        table_rows += f"<tr><td><strong>{anno}</strong></td>{cells}</tr>"
    # Riga totali
    total_cells = "".join(
        f"<td><strong>€ {r.get('riserva_totale', 0):,.0f}</strong></td>"
        for r in riserve_risultati
    )
    table_rows += f"<tr><td><strong>TOTALE</strong></td>{total_cells}</tr>"
    header_cells = "".join(f"<th>{r['metodo']}</th>" for r in riserve_risultati)

    sections.append(f"""
    <section>
      <h2><span class="section-num">1</span> Sintesi Riserve per Metodo</h2>
      <div class="kpi-grid">{kpi_cards}</div>
      <table class="report-table">
        <thead><tr><th>Anno</th>{header_cells}</tr></thead>
        <tbody>{table_rows}</tbody>
      </table>
    </section>
    """)

    # ── 2. Grafico confronto per anno ─────────────────────────────────────────
    fig_bar = go.Figure()
    for idx, r in enumerate(riserve_risultati):
        rpp = list(r.get("riserve_per_anno", []))
        fig_bar.add_trace(go.Bar(
            name=r["metodo"], x=anni_label, y=rpp,
            marker_color=PAL[idx % len(PAL)],
        ))
    fig_bar.update_layout(
        barmode="group",
        title="Riserva IBNR per Anno e Metodo",
        xaxis_title="Anno di Accadimento",
        yaxis_title="Riserva (€)",
        height=400, **DARK_LAYOUT,
    )
    img_bar = _fig_to_b64(fig_bar, height=400)

    fig_tot = go.Figure(go.Bar(
        x=[r["metodo"] for r in riserve_risultati],
        y=[r.get("riserva_totale", 0) for r in riserve_risultati],
        marker_color=PAL[:n_metodi],
        text=[f"€ {r.get('riserva_totale',0):,.0f}" for r in riserve_risultati],
        textposition="outside",
        textfont=dict(color="#e0f2f1", size=11),
    ))
    fig_tot.update_layout(
        title="Riserva Totale IBNR per Metodo",
        yaxis_title="€",
        height=380, **DARK_LAYOUT,
    )
    fig_tot.update_layout(yaxis=dict(
        **DARK_LAYOUT["yaxis"],
        range=[0, riserva_max * 1.18],
    ))
    img_tot = _fig_to_b64(fig_tot, height=380)

    sections.append(f"""
    <section>
      <h2><span class="section-num">2</span> Confronto Riserve per Anno</h2>
      <div class="chart-container">
        <img src="data:image/png;base64,{img_bar}"/>
        <div class="chart-caption">Confronto riserve IBNR per accident year e metodo</div>
      </div>
      <div class="chart-container">
        <img src="data:image/png;base64,{img_tot}"/>
        <div class="chart-caption">Riserva totale stimata per metodo</div>
      </div>
    </section>
    """)

    # ── 3. Analisi spread ─────────────────────────────────────────────────────
    spread_class = "alert-warning" if spread_pct > 20 else "alert-ok"
    spread_icon  = "⚠️" if spread_pct > 20 else "✅"
    spread_msg   = (
        "Spread elevato: le metodologie divergono significativamente. "
        "Analizzare le cause prima di selezionare la stima finale."
        if spread_pct > 20 else
        "Le stime sono ragionevolmente coerenti tra loro."
    )
    fill_pct = min(spread_pct, 100)

    metodi_sorted = sorted(riserve_risultati, key=lambda r: r.get("riserva_totale", 0))
    range_rows = "".join(f"""
      <tr>
        <td>{r['metodo']}</td>
        <td>€ {r.get('riserva_totale',0):,.0f}</td>
        <td>€ {r.get('riserva_totale',0) - riserva_min:,.0f}</td>
        <td>{(r.get('riserva_totale',0) - riserva_min)/riserva_max*100 if riserva_max>0 else 0:.1f}%</td>
      </tr>
    """ for r in metodi_sorted)

    sections.append(f"""
    <section>
      <h2><span class="section-num">3</span> Analisi dello Spread</h2>
      <div class="alert {spread_class}">
        <span class="alert-icon">{spread_icon}</span>
        <div>
          <strong>Range: € {riserva_min:,.0f} — € {riserva_max:,.0f}
          &nbsp;|&nbsp; Spread: € {riserva_max-riserva_min:,.0f} ({spread_pct:.1f}%)</strong><br/>
          {spread_msg}
        </div>
      </div>
      <div class="spread-bar"><div class="spread-fill" style="width:{fill_pct}%"></div></div>
      <table class="report-table">
        <thead><tr><th>Metodo</th><th>Riserva Totale</th><th>Δ dal minimo</th><th>% dal minimo</th></tr></thead>
        <tbody>{range_rows}</tbody>
      </table>
    </section>
    """)

    # ── 4. LDF selezionati (se disponibili) ───────────────────────────────────
    if ldf_selection is not None:
        ldf_rows = ""
        for _, row in ldf_selection.summary.iterrows():
            ldf_rows += "<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>"
        ldf_headers = "".join(f"<th>{c}</th>" for c in ldf_selection.summary.columns)

        notes_html = ""
        if ldf_selection.notes:
            items = "".join(f"<li>{n}</li>" for n in ldf_selection.notes)
            notes_html = f"<ul class='note-list'>{items}</ul>"
        else:
            notes_html = '<p class="alert alert-ok"><span class="alert-icon">✅</span>Nessuna criticità rilevata nella selezione LDF.</p>'

        sections.append(f"""
        <section>
          <h2><span class="section-num">4</span> LDF Selezionati</h2>
          <p>
            <strong>Metodo:</strong> {ldf_selection.method}
            &nbsp;·&nbsp;
            <strong>Outlier:</strong> {ldf_selection.outlier_method.upper()}
            &nbsp;·&nbsp;
            <strong>Anni esclusi:</strong>
            {[anni_label[i] for i in ldf_selection.excluded_years] or 'nessuno'}
          </p>
          <table class="report-table">
            <thead><tr>{ldf_headers}</tr></thead>
            <tbody>{ldf_rows}</tbody>
          </table>
          <h3>Note diagnostiche</h3>
          {notes_html}
        </section>
        """)

    # ── 5. Back-test ──────────────────────────────────────────────────────────
    if backtest_result is not None:
        n_diag = backtest_result.get("n_diagonali_rimosse", 1)
        bt_metodi = backtest_result.get("metodi", {})

        # Grafico errori %
        fig_err = go.Figure()
        for idx, (nome, dati) in enumerate(bt_metodi.items()):
            errs = [
                v if not np.isnan(v) else 0
                for v in dati["errori_pct"]
            ]
            fig_err.add_trace(go.Bar(
                name=nome, x=anni_label, y=errs,
                marker_color=PAL[idx % len(PAL)],
            ))
        fig_err.add_hline(y=0, line_dash="dash",
                          line_color="rgba(255,255,255,0.3)")
        fig_err.update_layout(
            barmode="group",
            title=f"Errore % per Anno e Metodo (diagonali rimosse: {n_diag})",
            xaxis_title="Anno di Accadimento",
            yaxis_title="Errore %",
            height=380, **DARK_LAYOUT,
        )
        img_err = _fig_to_b64(fig_err, height=380)

        # Tabella RMSE
        rmse_rows = ""
        for nome, dati in bt_metodi.items():
            err_abs = np.array(dati["errori_assoluti"])
            rmse = float(np.sqrt(np.nanmean(err_abs ** 2)))
            bias = float(np.nanmean(err_abs))
            rmse_rows += f"""
            <tr>
              <td>{nome}</td>
              <td>€ {rmse:,.0f}</td>
              <td>€ {bias:,.0f}</td>
            </tr>"""

        # Tabelle per metodo
        bt_tables = ""
        for nome, dati in bt_metodi.items():
            rows = ""
            for i, anno in enumerate(anni_label):
                ep = dati["errori_pct"][i]
                ep_str = f"{ep:.1f}%" if not np.isnan(ep) else "—"
                rows += f"""<tr>
                  <td>{anno}</td>
                  <td>€ {dati['ultimati_proiettati'][i]:,.0f}</td>
                  <td>€ {dati['ultimati_veri'][i]:,.0f}</td>
                  <td>€ {dati['errori_assoluti'][i]:,.0f}</td>
                  <td>{ep_str}</td>
                </tr>"""
            bt_tables += f"""
            <h3>{nome}</h3>
            <table class="report-table">
              <thead><tr>
                <th>Anno</th>
                <th>Ultimato proiettato</th>
                <th>Ultimato vero (proxy)</th>
                <th>Errore assoluto</th>
                <th>Errore %</th>
              </tr></thead>
              <tbody>{rows}</tbody>
            </table>"""

        sections.append(f"""
        <section>
          <h2><span class="section-num">{'5' if ldf_selection else '4'}</span>
              Back-test Retrospettivo</h2>
          <p>Diagonali rimosse: <strong>{n_diag}</strong>.
             L'ultimato "vero" è il proxy dal triangolo completo.</p>
          <div class="chart-container">
            <img src="data:image/png;base64,{img_err}"/>
            <div class="chart-caption">
              Errore % per anno e metodo — positivo = sovrastima, negativo = sottostima
            </div>
          </div>
          <h3>RMSE per metodo</h3>
          <table class="report-table">
            <thead><tr><th>Metodo</th><th>RMSE (€)</th><th>Bias medio (€)</th></tr></thead>
            <tbody>{rmse_rows}</tbody>
          </table>
          {bt_tables}
        </section>
        """)

    # ── Footer ────────────────────────────────────────────────────────────────
    sections.append(f"""
    <div class="report-footer">
      Generato il {date.today().strftime("%d/%m/%Y")} &nbsp;·&nbsp;
      Insurance Claims Reserve &nbsp;·&nbsp;
      Friedland – Estimating Unpaid Claims Using Basic Techniques (CAS, 2010)
    </div>
    """)

    html = f"""<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{titolo}</title>
  {css}
</head>
<body>{''.join(sections)}</body>
</html>"""

    html_bytes = html.encode("utf-8")

    pdf_bytes: Optional[bytes] = None
    if produrre_pdf:
        try:
            from weasyprint import HTML as WeasyprintHTML
            pdf_bytes = WeasyprintHTML(string=html).write_pdf()
        except ImportError:
            warnings.warn("weasyprint non installato.", stacklevel=2)
        except Exception as exc:
            warnings.warn(f"Generazione PDF fallita: {exc}", stacklevel=2)

    return html_bytes, pdf_bytes
# ══════════════════════════════════════════════════════════════════════════════
#  ESEMPIO MINIMO DI UTILIZZO
# ══════════════════════════════════════════════════════════════════════════════
#
#  import numpy as np
#  from utils.diagnostica import select_ldf, detect_anomalies, generate_evaluation_report
#
#  triangle = np.array([
#      [210_000, 330_000, 352_000, 361_000, 365_000],
#      [195_000, 310_000, 340_000, 351_000,     np.nan],
#      [220_000, 345_000, 370_000,     np.nan, np.nan],
#      [230_000, 360_000,     np.nan, np.nan, np.nan],
#      [240_000,     np.nan, np.nan, np.nan, np.nan],
#  ])
#  anni = ["2019", "2020", "2021", "2022", "2023"]
#
#  sel = select_ldf(triangle, anni, method="weighted", outlier_method="iqr",
#                   remove_outliers=True, exclude_years=[1])  # esclude 2020
#  print(sel.summary)
#  print(sel.notes)
#
#  anom = detect_anomalies(triangle, anni)
#  html_bytes, pdf_bytes = generate_evaluation_report(
#      triangle, anni, sel, anomalie=anom, produrre_pdf=False
#  )
#  with open("report.html", "wb") as f:
#      f.write(html_bytes)
