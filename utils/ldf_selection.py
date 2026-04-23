"""
ldf_selection.py
────────────────
Selezione attuariale degli LDF con diagnostica, gestione outlier e giudizio.

Dipende da riserva_sinistri (age_to_age_matrix, compute_factors) ma non
modifica nulla di quel modulo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from .riserva_sinistri import age_to_age_matrix, compute_factors


# ══════════════════════════════════════════════════════════════════════════════
#  MEDIE ELEMENTARI
# ══════════════════════════════════════════════════════════════════════════════

def _volume_weighted(tri: np.ndarray, col: int, mask: np.ndarray) -> float:
    """Volume-weighted average per la colonna col, usando solo le righe in mask."""
    rows = np.where(mask)[0]
    num = sum(tri[r, col + 1] for r in rows
              if not np.isnan(tri[r, col + 1]) and tri[r, col] > 0)
    den = sum(tri[r, col] for r in rows
              if not np.isnan(tri[r, col + 1]) and tri[r, col] > 0)
    return num / den if den > 0 else 1.0


def _simple_mean(ratios: np.ndarray) -> float:
    return float(np.nanmean(ratios)) if len(ratios) > 0 else 1.0


def _trimmed_mean(ratios: np.ndarray, pct: float = 0.10) -> float:
    """Media tagliata simmetrica (rimuove pct dal basso e dall'alto)."""
    if len(ratios) < 3:
        return _simple_mean(ratios)
    k = max(1, int(len(ratios) * pct))
    sorted_r = np.sort(ratios)
    return float(np.mean(sorted_r[k:-k]))


def _compute_all_averages(
    ldf_matrix: np.ndarray,
    triangle: np.ndarray,
    mask: np.ndarray,
) -> pd.DataFrame:
    """
    Calcola all-years, ultimi-3, ultimi-5, volume-weighted e trimmed
    per ogni periodo di sviluppo.
    Restituisce un DataFrame (5 × n_periodi).
    """
    n_dev = ldf_matrix.shape[1]
    records = []
    labels = ["All-Years", "Ultimi 3", "Ultimi 5", "Volume-Weighted", "Trimmed (10%)"]

    for col in range(n_dev):
        row_vals = ldf_matrix[mask, col]
        valid = row_vals[~np.isnan(row_vals)]

        last3 = valid[-3:] if len(valid) >= 3 else valid
        last5 = valid[-5:] if len(valid) >= 5 else valid

        # Volume-weighted: richiede accesso al triangolo originale
        rows_ok = np.where(mask)[0]
        n_rows_avail = triangle.shape[0] - 1 - col
        rows_for_vol = np.array([r for r in rows_ok if r < n_rows_avail])
        vw_mask = np.zeros(triangle.shape[0], dtype=bool)
        vw_mask[rows_for_vol] = True

        vw = _volume_weighted(triangle, col, vw_mask) if vw_mask.any() else 1.0

        records.append({
            "Sviluppo": f"{col+1}→{col+2}",
            "All-Years":       _simple_mean(valid),
            "Ultimi 3":        _simple_mean(last3),
            "Ultimi 5":        _simple_mean(last5),
            "Volume-Weighted": vw,
            "Trimmed (10%)":   _trimmed_mean(valid),
            "N. osservazioni": int((~np.isnan(ldf_matrix[mask, col])).sum()),
            "CoV":             float(np.nanstd(valid) / np.nanmean(valid))
                               if np.nanmean(valid) > 0 and len(valid) > 1 else 0.0,
        })

    return pd.DataFrame(records).set_index("Sviluppo")


# ══════════════════════════════════════════════════════════════════════════════
#  OUTLIER DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_outliers(
    ldf_matrix: np.ndarray,
    method: str = "iqr",
    threshold: float = 1.5,
) -> dict[str, list[tuple[int, int, float]]]:
    """
    Identifica outlier nella matrice dei link ratio.

    Parameters
    ----------
    method    : 'iqr' oppure 'zscore'
    threshold : moltiplicatore IQR (default 1.5) o soglia Z (default 3.0)

    Returns
    -------
    dict { "outliers": [(row, col, value), ...], "metodo": str }
    """
    n_dev = ldf_matrix.shape[1]
    outliers = []

    for col in range(n_dev):
        vals = ldf_matrix[:, col]
        valid_idx = np.where(~np.isnan(vals))[0]
        valid_vals = vals[valid_idx]
        if len(valid_vals) < 3:
            continue

        if method == "iqr":
            q1, q3 = np.percentile(valid_vals, [25, 75])
            iqr = q3 - q1
            lo, hi = q1 - threshold * iqr, q3 + threshold * iqr
            flag = (valid_vals < lo) | (valid_vals > hi)
        else:  # zscore
            mu, sigma = np.mean(valid_vals), np.std(valid_vals)
            flag = np.abs((valid_vals - mu) / sigma) > threshold if sigma > 0 \
                   else np.zeros(len(valid_vals), dtype=bool)

        for i, f in zip(valid_idx[flag], valid_vals[flag]):
            outliers.append((int(i), int(col), float(f)))

    return {"outliers": outliers, "metodo": method, "threshold": threshold}


# ══════════════════════════════════════════════════════════════════════════════
#  FUNZIONE PRINCIPALE: select_ldf
# ══════════════════════════════════════════════════════════════════════════════

def select_ldf(
    triangle: np.ndarray,
    accident_years: Optional[list] = None,
    method: str = "weighted",
    exclude_years: Optional[list] = None,
    remove_outliers: bool = True,
    outlier_method: str = "iqr",
    outlier_threshold: float = 1.5,
    tail_factor: float = 1.0,
) -> dict:
    """
    Selezione attuariale degli LDF con giudizio strutturato.

    Parameters
    ----------
    triangle         : triangolo cumulato n×n
    accident_years   : lista anni accadimento (es. [2016, ..., 2024])
    method           : 'weighted' | 'simple' | 'last3' | 'last5' | 'trimmed'
    exclude_years    : anni accadimento da escludere (es. [2020] per Covid)
    remove_outliers  : se True, esclude i ratios identificati come outlier
                       prima di calcolare il metodo scelto
    outlier_method   : 'iqr' | 'zscore'
    outlier_threshold: soglia per il rilevamento outlier
    tail_factor      : fattore di coda da applicare sull'ultimo LDF

    Returns
    -------
    {
      "ldf_selezionati"  : np.ndarray,
      "cdf_to_ultimate"  : np.ndarray,
      "riepilogo_medie"  : pd.DataFrame,
      "outlier_info"     : dict,
      "giudizio"         : dict   ← spiegazione strutturata della selezione
    }
    """
    n = triangle.shape[0]
    if accident_years is None:
        accident_years = list(range(1, n + 1))

    # ── Maschera anni da includere ──────────────────────────────────────────
    exclude_set = set(exclude_years or [])
    year_mask = np.array([y not in exclude_set for y in accident_years])

    # ── Matrice link ratio (tutte le celle, inclusi anni esclusi) ───────────
    ldf_matrix_full = age_to_age_matrix(triangle)

    # ── Outlier detection sulla matrice piena ────────────────────────────────
    outlier_info = detect_outliers(ldf_matrix_full, outlier_method, outlier_threshold)

    # ── Matrice "pulita" per la selezione: applica maschera anni + outlier ──
    ldf_matrix_clean = ldf_matrix_full.copy()
    ldf_matrix_clean[~year_mask, :] = np.nan          # rimuovi anni esclusi

    if remove_outliers:
        for row, col, _ in outlier_info["outliers"]:
            if year_mask[row]:                         # solo se non già escluso
                ldf_matrix_clean[row, col] = np.nan

    # ── Calcola tutte le medie ────────────────────────────────────────────────
    riepilogo = _compute_all_averages(ldf_matrix_clean, triangle, year_mask)

    # ── Selezione del metodo ─────────────────────────────────────────────────
    _METHOD_MAP = {
        "weighted": "Volume-Weighted",
        "simple":   "All-Years",
        "last3":    "Ultimi 3",
        "last5":    "Ultimi 5",
        "trimmed":  "Trimmed (10%)",
    }
    col_selected = _METHOD_MAP.get(method, "Volume-Weighted")
    ldf_selezionati = riepilogo[col_selected].values.copy()

    # Applica tail factor sull'ultimo periodo
    ldf_selezionati[-1] *= tail_factor

    # ── CDF to ultimate ───────────────────────────────────────────────────────
    cdf = np.ones(n)
    cdf[-1] = 1.0
    for i in range(n - 2, -1, -1):
        cdf[i] = cdf[i + 1] * ldf_selezionati[i]

    # ── Criticità: alta variabilità ───────────────────────────────────────────
    criticita = []
    HIGH_COV = 0.15
    for dev, cov in zip(riepilogo.index, riepilogo["CoV"]):
        if cov > HIGH_COV:
            criticita.append(f"Alta variabilità ({cov:.1%}) nel periodo {dev}")

    n_outliers = len(outlier_info["outliers"])
    if n_outliers > 0:
        criticita.append(f"Rilevati {n_outliers} outlier con metodo {outlier_method.upper()}")

    # ── Giudizio strutturato ──────────────────────────────────────────────────
    giudizio = {
        "metodo_scelto":     col_selected,
        "anni_esclusi":      list(exclude_set),
        "outlier_rimossi":   remove_outliers,
        "n_outlier":         n_outliers,
        "outlier_dettaglio": [
            {"anno_accadimento": accident_years[r], "sviluppo": f"{c+1}→{c+2}", "valore": round(v, 4)}
            for r, c, v in outlier_info["outliers"]
        ],
        "tail_factor":       tail_factor,
        "criticita":         criticita if criticita else ["Nessuna criticità rilevata."],
    }

    return {
        "ldf_selezionati":  ldf_selezionati,
        "cdf_to_ultimate":  cdf,
        "riepilogo_medie":  riepilogo,
        "ldf_matrix_full":  ldf_matrix_full,
        "outlier_info":     outlier_info,
        "giudizio":         giudizio,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ESEMPIO MINIMO DI UTILIZZO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Triangolo 6×6 di esempio (cumulato, NaN per celle future)
    T = np.array([
        [1_000, 1_500, 1_750, 1_850, 1_900, 1_920],
        [1_100, 1_650, 1_900, 2_000, 2_050,  np.nan],
        [1_200, 1_780, 2_050, 2_150,  np.nan,  np.nan],
        [1_300, 1_900, 2_200,  np.nan,  np.nan,  np.nan],
        [1_400, 2_050,  np.nan,  np.nan,  np.nan,  np.nan],
        [1_500,  np.nan,  np.nan,  np.nan,  np.nan,  np.nan],
    ], dtype=float)

    anni = list(range(2018, 2024))

    result = select_ldf(
        triangle=T,
        accident_years=anni,
        method="weighted",
        exclude_years=[2020],
        remove_outliers=True,
    )

    print("LDF selezionati:")
    print(result["ldf_selezionati"].round(4))
    print("\nRiepilogo medie:")
    print(result["riepilogo_medie"].round(4))
    print("\nGiudizio attuariale:")
    for k, v in result["giudizio"].items():
        print(f"  {k}: {v}")
