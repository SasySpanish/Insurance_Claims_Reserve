"""
Modulo calcolo riserva sinistri.
Metodi implementati:
  1. Chain Ladder (sviluppo pagamenti)
  2. Bornhuetter-Ferguson
  3. Cape Cod
  4. Average Cost per Claim (ACPC)
"""

import numpy as np
import pandas as pd
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: triangolo di sviluppo
# ══════════════════════════════════════════════════════════════════════════════

def _validate_triangle(triangle: np.ndarray) -> None:
    n = triangle.shape[0]
    if triangle.shape[1] != n:
        raise ValueError("Il triangolo deve essere quadrato (n × n).")


def _age_to_age_factors(triangle: np.ndarray) -> np.ndarray:
    """Calcola i fattori età-età (link ratios) dal triangolo."""
    n = triangle.shape[0]
    factors = []
    for col in range(n - 1):
        num = sum(
            triangle[row, col + 1]
            for row in range(n - 1 - col)
            if triangle[row, col] > 0
        )
        den = sum(
            triangle[row, col]
            for row in range(n - 1 - col)
            if triangle[row, col] > 0
        )
        factors.append(num / den if den > 0 else 1.0)
    return np.array(factors)


def build_development_triangle(pagamenti: pd.DataFrame) -> np.ndarray:
    """
    Costruisce un triangolo di sviluppo cumulato da un DataFrame
    con colonne: anno_accadimento, anno_sviluppo, pagato.
    """
    pivot = pagamenti.pivot_table(
        index="anno_accadimento",
        columns="anno_sviluppo",
        values="pagato",
        aggfunc="sum",
        fill_value=0
    )
    # Cumula per righe
    cumulative = pivot.cumsum(axis=1).values.astype(float)
    # Metti NaN nella parte superiore destra (dati futuri)
    n = cumulative.shape[0]
    m = cumulative.shape[1]
    for i in range(n):
        for j in range(m):
            if i + j >= n:
                cumulative[i, j] = np.nan
    return cumulative, pivot.index.tolist(), pivot.columns.tolist()


# ══════════════════════════════════════════════════════════════════════════════
#  1. CHAIN LADDER
# ══════════════════════════════════════════════════════════════════════════════

def chain_ladder(triangle: np.ndarray) -> dict:
    """
    Metodo Chain Ladder standard.
    Input: triangolo cumulato n×n (NaN per celle future).
    Output: triangolo completato, fattori di sviluppo, riserve per anno.
    """
    n = triangle.shape[0]
    tri = triangle.copy().astype(float)

    # Fattori età-età dall'ultima diagonale disponibile
    factors = []
    for col in range(n - 1):
        nums, dens = [], []
        for row in range(n):
            if col + 1 < n and not np.isnan(tri[row, col]) and not np.isnan(tri[row, col + 1]):
                dens.append(tri[row, col])
                nums.append(tri[row, col + 1])
        f = sum(nums) / sum(dens) if sum(dens) > 0 else 1.0
        factors.append(f)

    # Sviluppa il triangolo
    for row in range(n):
        for col in range(n - 1):
            if np.isnan(tri[row, col + 1]) and not np.isnan(tri[row, col]):
                tri[row, col + 1] = tri[row, col] * factors[col]

    # Diagonale attuale (ultimi pagati noti)
    diagonale = np.array([
        triangle[i, n - 1 - i] if not np.isnan(triangle[i, n - 1 - i]) else 0.0
        for i in range(n)
    ])
    ultimati = tri[:, n - 1]
    riserve = np.maximum(ultimati - diagonale, 0.0)

    return {
        "metodo": "Chain Ladder",
        "fattori_sviluppo": factors,
        "triangolo_completato": tri,
        "ultimati": ultimati,
        "pagati_attuali": diagonale,
        "riserve_per_anno": riserve,
        "riserva_totale": riserve.sum(),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  2. BORNHUETTER-FERGUSON
# ══════════════════════════════════════════════════════════════════════════════

def bornhuetter_ferguson(
    triangle: np.ndarray,
    premi_di_rischio: np.ndarray,
    loss_ratio_atteso: float,
) -> dict:
    """
    Metodo Bornhuetter-Ferguson.
    Combina esperienza storica (CL) con a-priori (LR × premio).
    
    premi_di_rischio: array di n premi, uno per anno di accadimento.
    loss_ratio_atteso: loss ratio a priori (es. 0.65 per 65%).
    """
    n = triangle.shape[0]
    cl = chain_ladder(triangle)
    factors = cl["fattori_sviluppo"]

    # Fattori di sviluppo cumulativi (CDF tail-to-ult)
    cdfs = np.ones(n)
    for i in range(n - 1, 0, -1):
        cdfs[i - 1] = cdfs[i] * factors[i - 1] if i - 1 < len(factors) else cdfs[i]
    # % già sviluppato per ogni anno
    pct_sviluppato = 1.0 / cdfs

    # Perdita a priori per anno
    perdita_apriori = premi_di_rischio * loss_ratio_atteso

    # Diagonale attuale
    diagonale = np.array([
        triangle[i, n - 1 - i] if not np.isnan(triangle[i, n - 1 - i]) else 0.0
        for i in range(n)
    ])

    # BF: ultimato = pagato + (1 - % sviluppato) × perdita_apriori
    ultimati_bf = diagonale + (1 - pct_sviluppato) * perdita_apriori
    riserve_bf = np.maximum(ultimati_bf - diagonale, 0.0)

    return {
        "metodo": "Bornhuetter-Ferguson",
        "loss_ratio_atteso": loss_ratio_atteso,
        "cdfs": cdfs,
        "pct_sviluppato": pct_sviluppato,
        "perdita_apriori": perdita_apriori,
        "pagati_attuali": diagonale,
        "ultimati": ultimati_bf,
        "riserve_per_anno": riserve_bf,
        "riserva_totale": riserve_bf.sum(),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  3. CAPE COD
# ══════════════════════════════════════════════════════════════════════════════

def cape_cod(
    triangle: np.ndarray,
    premi_di_rischio: np.ndarray,
) -> dict:
    """
    Metodo Cape Cod.
    Stima il loss ratio a priori direttamente dai dati osservati.
    ELR = Σ pagati / Σ (premi × % sviluppato)
    """
    n = triangle.shape[0]
    cl = chain_ladder(triangle)
    factors = cl["fattori_sviluppo"]

    cdfs = np.ones(n)
    for i in range(n - 1, 0, -1):
        cdfs[i - 1] = cdfs[i] * factors[i - 1] if i - 1 < len(factors) else cdfs[i]
    pct_sviluppato = 1.0 / cdfs

    diagonale = np.array([
        triangle[i, n - 1 - i] if not np.isnan(triangle[i, n - 1 - i]) else 0.0
        for i in range(n)
    ])

    # ELR empirico
    num = diagonale.sum()
    den = (premi_di_rischio * pct_sviluppato).sum()
    elr = num / den if den > 0 else 0.0

    perdita_apriori = premi_di_rischio * elr
    ultimati_cc = diagonale + (1 - pct_sviluppato) * perdita_apriori
    riserve_cc = np.maximum(ultimati_cc - diagonale, 0.0)

    return {
        "metodo": "Cape Cod",
        "elr_stimato": elr,
        "cdfs": cdfs,
        "pct_sviluppato": pct_sviluppato,
        "perdita_apriori": perdita_apriori,
        "pagati_attuali": diagonale,
        "ultimati": ultimati_cc,
        "riserve_per_anno": riserve_cc,
        "riserva_totale": riserve_cc.sum(),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  4. AVERAGE COST PER CLAIM (ACPC)
# ══════════════════════════════════════════════════════════════════════════════

def average_cost_per_claim(
    numero_sinistri: np.ndarray,      # sinistri IBNR stimati per anno
    costo_medio_sinistro: float,       # costo medio per sinistro
    fattore_inflazione: float = 0.0,   # % inflazione annua
    anni_accadimento: Optional[list] = None,
    anno_valutazione: Optional[int] = None,
) -> dict:
    """
    Metodo Average Cost per Claim.
    Riserva = N° sinistri IBNR × costo medio × (1 + inflazione)^t
    """
    n = len(numero_sinistri)
    if anni_accadimento is None:
        anni_accadimento = list(range(1, n + 1))
    if anno_valutazione is None:
        anno_valutazione = max(anni_accadimento)

    riserve = []
    costi_inflazionati = []
    for i, anno in enumerate(anni_accadimento):
        t = anno_valutazione - anno
        costo_inf = costo_medio_sinistro * ((1 + fattore_inflazione / 100) ** t)
        riserva_anno = numero_sinistri[i] * costo_inf
        riserve.append(riserva_anno)
        costi_inflazionati.append(costo_inf)

    riserve = np.array(riserve)
    return {
        "metodo": "Average Cost per Claim",
        "numero_sinistri_ibnr": numero_sinistri,
        "costo_medio_base": costo_medio_sinistro,
        "fattore_inflazione_perc": fattore_inflazione,
        "costi_inflazionati": np.array(costi_inflazionati),
        "riserve_per_anno": riserve,
        "riserva_totale": riserve.sum(),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: tabella riepilogativa riserve
# ══════════════════════════════════════════════════════════════════════════════

def tabella_riepilogo_riserve(
    anni: list,
    risultati: list[dict],
) -> pd.DataFrame:
    """Crea una tabella comparativa tra più metodi di riserva."""
    df = pd.DataFrame({"Anno Accadimento": anni})
    for res in risultati:
        metodo = res["metodo"]
        riserve = res["riserve_per_anno"]
        ultimati = res.get("ultimati", [None] * len(anni))
        df[f"Riserva {metodo} (€)"] = riserve
        if all(v is not None for v in ultimati):
            df[f"Ultimato {metodo} (€)"] = ultimati
    # Totali
    totale_row = {"Anno Accadimento": "TOTALE"}
    for col in df.columns[1:]:
        totale_row[col] = df[col].sum()
    df = pd.concat([df, pd.DataFrame([totale_row])], ignore_index=True)
    return df
