"""
Modulo calcolo riserva sinistri.
Metodi implementati:
  1. Chain Ladder (con selezione fattori, tipi di media)
  2. Bornhuetter-Ferguson
  3. Cape Cod
  4. Average Cost per Claim (ACPC) con stima conteggi da triangolo
  5. Case Outstanding Development (Wiser Approach #1)
  6. Evaluation / Back-test retrospettivo
"""

import numpy as np
import pandas as pd
from typing import Optional


# ═════════════════════════════════════════════════════════════════════════════
#  VALIDAZIONE
# ══════════════════════════════════════════════════════════════════════════════

def validate_triangle(triangle: np.ndarray) -> list[str]:
    """
    Controlla il triangolo e restituisce una lista di warning (stringhe).
    Non lancia eccezioni: è l'interfaccia a decidere come mostrarli.
    """
    warnings = []
    n = triangle.shape[0]
    if triangle.shape[1] != n:
        warnings.append("Il triangolo non è quadrato (n×n).")
    # Valori negativi
    valid = triangle[~np.isnan(triangle)]
    if np.any(valid < 0):
        warnings.append("Presenti valori negativi nel triangolo.")
    # Verifica cumulazione: ogni riga deve essere non decrescente
    for i in range(n):
        row = [triangle[i, j] for j in range(n - i) if not np.isnan(triangle[i, j])]
        for k in range(1, len(row)):
            if row[k] < row[k - 1]:
                warnings.append(
                    f"Anno {i+1}: i pagamenti cumulati decrescono alla colonna {k+1} "
                    f"({row[k]:,.0f} < {row[k-1]:,.0f}). Verifica che i dati siano cumulati."
                )
                break
    # Diagonale principale: almeno tutti non-NaN
    for i in range(n):
        j = n - 1 - i
        if np.isnan(triangle[i, j]):
            warnings.append(f"Anno {i+1}: manca il valore sulla diagonale corrente (colonna {j+1}).")
    return warnings


def _diagonale_attuale(triangle: np.ndarray) -> np.ndarray:
    """Estrae l'ultima diagonale osservata."""
    n = triangle.shape[0]
    return np.array([
        triangle[i, n - 1 - i] if not np.isnan(triangle[i, n - 1 - i]) else 0.0
        for i in range(n)
    ])


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: triangolo di sviluppo
# ══════════════════════════════════════════════════════════════════════════════

def build_development_triangle(pagamenti: pd.DataFrame):
    """
    Costruisce un triangolo di sviluppo cumulato da un DataFrame
    con colonne: anno_accadimento, anno_sviluppo, pagato.
    """
    pivot = pagamenti.pivot_table(
        index="anno_accadimento",
        columns="anno_sviluppo",
        values="pagato",
        aggfunc="sum",
        fill_value=0,
    )
    cumulative = pivot.cumsum(axis=1).values.astype(float)
    n = cumulative.shape[0]
    m = cumulative.shape[1]
    for i in range(n):
        for j in range(m):
            if i + j >= n:
                cumulative[i, j] = np.nan
    return cumulative, pivot.index.tolist(), pivot.columns.tolist()


# ══════════════════════════════════════════════════════════════════════════════
#  FATTORI ETÀ-ETÀ con più tipi di media
# ══════════════════════════════════════════════════════════════════════════════

def age_to_age_matrix(triangle: np.ndarray) -> np.ndarray:
    """
    Restituisce la matrice (n-1) × (n-1) dei singoli link ratio età-età.
    Cella [i, j] = triangle[i, j+1] / triangle[i, j] oppure NaN se non disponibile.
    """
    n = triangle.shape[0]
    mat = np.full((n, n - 1), np.nan)
    for col in range(n - 1):
        for row in range(n - 1 - col):
            if triangle[row, col] > 0 and not np.isnan(triangle[row, col + 1]):
                mat[row, col] = triangle[row, col + 1] / triangle[row, col]
    return mat


def compute_factors(triangle: np.ndarray, tipo_media: str = "volume") -> np.ndarray:
    """
    Calcola i fattori età-età aggregati.
    tipo_media: 'volume' (volume-weighted), 'semplice' (simple average),
                'mediana', 'ultimi3' (vol-weighted ultimi 3 anni).
    """
    n = triangle.shape[0]
    factors = []
    for col in range(n - 1):
        rows_avail = n - 1 - col
        if tipo_media == "volume":
            num = sum(triangle[r, col + 1] for r in range(rows_avail)
                      if not np.isnan(triangle[r, col + 1]) and triangle[r, col] > 0)
            den = sum(triangle[r, col] for r in range(rows_avail)
                      if not np.isnan(triangle[r, col + 1]) and triangle[r, col] > 0)
            f = num / den if den > 0 else 1.0
        elif tipo_media == "semplice":
            ratios = [triangle[r, col + 1] / triangle[r, col]
                      for r in range(rows_avail)
                      if not np.isnan(triangle[r, col + 1]) and triangle[r, col] > 0]
            f = float(np.mean(ratios)) if ratios else 1.0
        elif tipo_media == "mediana":
            ratios = [triangle[r, col + 1] / triangle[r, col]
                      for r in range(rows_avail)
                      if not np.isnan(triangle[r, col + 1]) and triangle[r, col] > 0]
            f = float(np.median(ratios)) if ratios else 1.0
        elif tipo_media == "ultimi3":
            k = min(3, rows_avail)
            num = sum(triangle[r, col + 1] for r in range(k)
                      if not np.isnan(triangle[r, col + 1]) and triangle[r, col] > 0)
            den = sum(triangle[r, col] for r in range(k)
                      if not np.isnan(triangle[r, col + 1]) and triangle[r, col] > 0)
            f = num / den if den > 0 else 1.0
        else:
            f = 1.0
        factors.append(f)
    return np.array(factors)


def cdfs_from_factors(factors: np.ndarray) -> np.ndarray:
    """Calcola i CDF cumulativi (tail-to-ultimate) dai fattori età-età."""
    n = len(factors) + 1
    cdfs = np.ones(n)
    for i in range(n - 2, -1, -1):
        cdfs[i] = cdfs[i + 1] * factors[i]
    return cdfs


# ══════════════════════════════════════════════════════════════════════════════
#  1. CHAIN LADDER
# ══════════════════════════════════════════════════════════════════════════════

def chain_ladder(
    triangle: np.ndarray,
    tipo_media: str = "volume",
    fattori_manuali: Optional[np.ndarray] = None,
    tail_factor: float = 1.0,
) -> dict:
    """
    Metodo Chain Ladder standard con supporto a media selezionabile,
    fattori manuali e tail factor.

    Parameters
    ----------
    triangle        : triangolo cumulato n×n (NaN per celle future).
    tipo_media      : 'volume' | 'semplice' | 'mediana' | 'ultimi3'
    fattori_manuali : se fornito, sovrascrive i fattori calcolati (array n-1).
    tail_factor     : fattore di coda (default 1.0 = nessuna coda).
    """
    n = triangle.shape[0]
    tri = triangle.copy().astype(float)

    # Fattori età-età
    if fattori_manuali is not None and len(fattori_manuali) == n - 1:
        factors = np.array(fattori_manuali, dtype=float)
    else:
        factors = compute_factors(tri, tipo_media)

    # Applica tail factor all'ultimo fattore cumulativo
    cdfs = cdfs_from_factors(factors)
    cdfs[0] = cdfs[0] * tail_factor  # tail applicato al CDF complessivo

    # Tutti i possibili link ratio singoli per audit
    ldf_matrix = age_to_age_matrix(tri)

    # Sviluppa il triangolo cella per cella
    for row in range(n):
        for col in range(n - 1):
            if np.isnan(tri[row, col + 1]) and not np.isnan(tri[row, col]):
                tri[row, col + 1] = tri[row, col] * factors[col]

    diagonale = _diagonale_attuale(triangle)
    ultimati = tri[:, n - 1] * tail_factor  # applica tail all'ultima colonna
    # Correggi: anni già maturi (ultima col già osservata) non devono avere tail doppio
    # In pratica: solo l'anno più immaturo (row=n-1) ha colonna già completa senza bisogno
    # del tail; lo applichiamo solo se non è già sulla colonna finale osservata.
    ultimati = np.array([
        tri[i, n - 1] * (tail_factor if (n - 1 - i) < n - 1 else 1.0)
        for i in range(n)
    ])
    # Semplificazione corretta: il tail si applica a tutti gli ultimati (column n-1 è sempre proiettata)
    ultimati = tri[:, n - 1] * tail_factor

    riserve = np.maximum(ultimati - diagonale, 0.0)

    return {
        "metodo": "Chain Ladder",
        "tipo_media": tipo_media,
        "tail_factor": tail_factor,
        "fattori_sviluppo": factors,
        "ldf_matrix": ldf_matrix,
        "cdfs": cdfs,
        "triangolo_completato": tri,
        "ultimati": ultimati,
        "pagati_attuali": diagonale,
        "riserve_per_anno": riserve,
        "riserva_totale": float(riserve.sum()),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  2. BORNHUETTER-FERGUSON
# ══════════════════════════════════════════════════════════════════════════════

def bornhuetter_ferguson(
    triangle: np.ndarray,
    premi_di_rischio: np.ndarray,
    loss_ratio_atteso: float,
    tipo_media: str = "volume",
    tail_factor: float = 1.0,
) -> dict:
    """
    Metodo Bornhuetter-Ferguson.
    Combina esperienza storica (CL) con a-priori (LR × premio).
    Restituisce anche l'ELR Cape Cod per confronto diagnostico.
    """
    n = triangle.shape[0]
    cl = chain_ladder(triangle, tipo_media=tipo_media, tail_factor=tail_factor)
    cdfs = cl["cdfs"]
    pct_sviluppato = 1.0 / cdfs

    perdita_apriori = premi_di_rischio * loss_ratio_atteso
    diagonale = _diagonale_attuale(triangle)

    ultimati_bf = diagonale + (1 - pct_sviluppato) * perdita_apriori
    riserve_bf = np.maximum(ultimati_bf - diagonale, 0.0)

    # ELR Cape Cod per diagnostica
    den_cc = (premi_di_rischio * pct_sviluppato).sum()
    elr_cape_cod = diagonale.sum() / den_cc if den_cc > 0 else 0.0
    divergenza_elr = abs(loss_ratio_atteso - elr_cape_cod) / elr_cape_cod if elr_cape_cod > 0 else 0.0

    return {
        "metodo": "Bornhuetter-Ferguson",
        "loss_ratio_atteso": loss_ratio_atteso,
        "elr_cape_cod_confronto": elr_cape_cod,
        "divergenza_elr_pct": divergenza_elr * 100,
        "cdfs": cdfs,
        "pct_sviluppato": pct_sviluppato,
        "perdita_apriori": perdita_apriori,
        "pagati_attuali": diagonale,
        "ultimati": ultimati_bf,
        "riserve_per_anno": riserve_bf,
        "riserva_totale": float(riserve_bf.sum()),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  3. CAPE COD
# ══════════════════════════════════════════════════════════════════════════════

def cape_cod(
    triangle: np.ndarray,
    premi_di_rischio: np.ndarray,
    tipo_media: str = "volume",
    tail_factor: float = 1.0,
) -> dict:
    """
    Metodo Cape Cod (Stanard-Buhlmann).
    ELR = Σ pagati / Σ (premi × % sviluppato)
    """
    n = triangle.shape[0]
    cl = chain_ladder(triangle, tipo_media=tipo_media, tail_factor=tail_factor)
    cdfs = cl["cdfs"]
    pct_sviluppato = 1.0 / cdfs
    diagonale = _diagonale_attuale(triangle)

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
        "riserva_totale": float(riserve_cc.sum()),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  4. AVERAGE COST PER CLAIM (ACPC) — con stima conteggi opzionale da triangolo
# ══════════════════════════════════════════════════════════════════════════════

def stima_conteggi_da_triangolo(
    triangle_counts: np.ndarray,
    tipo_media: str = "volume",
) -> np.ndarray:
    """
    Se disponibile il triangolo dei conteggi, proietta i conti ultimati
    e restituisce i sinistri IBNR per anno (ultimati - riportati).
    """
    cl = chain_ladder(triangle_counts, tipo_media=tipo_media)
    diag = _diagonale_attuale(triangle_counts)
    return np.maximum(cl["ultimati"] - diag, 0.0)


def average_cost_per_claim(
    numero_sinistri: np.ndarray,
    costo_medio_sinistro: float,
    fattore_inflazione: float = 0.0,
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

    riserve, costi_inflazionati = [], []
    for i, anno in enumerate(anni_accadimento):
        t = max(0, anno_valutazione - anno)
        costo_inf = costo_medio_sinistro * ((1 + fattore_inflazione / 100) ** t)
        riserve.append(numero_sinistri[i] * costo_inf)
        costi_inflazionati.append(costo_inf)

    riserve = np.array(riserve)
    return {
        "metodo": "Average Cost per Claim",
        "numero_sinistri_ibnr": numero_sinistri,
        "costo_medio_base": costo_medio_sinistro,
        "fattore_inflazione_perc": fattore_inflazione,
        "costi_inflazionati": np.array(costi_inflazionati),
        "riserve_per_anno": riserve,
        "riserva_totale": float(riserve.sum()),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  5. CASE OUTSTANDING DEVELOPMENT (Wiser Approach #1)
# ══════════════════════════════════════════════════════════════════════════════

def case_outstanding_development(
    triangle_paid: np.ndarray,
    triangle_case: np.ndarray,
    tipo_media: str = "volume",
) -> dict:
    """
    Case Outstanding Development Technique (Wiser).
    
    Parametri
    ---------
    triangle_paid : triangolo pagamenti cumulati (n×n, NaN per il futuro).
    triangle_case : triangolo riserve di testa (case outstanding) (n×n, NaN per il futuro).
                    Deve avere la stessa dimensione di triangle_paid.
    tipo_media    : tipo di media per aggregare i ratios ('volume', 'semplice', 'mediana', 'ultimi3').

    Logica
    ------
    Per ogni colonna j:
        ratio[i,j] = incremental_paid[i, j] / case_outstanding[i, j-1]
    dove incremental_paid[i,j] = paid[i,j] - paid[i,j-1].

    Poi si seleziona un fattore medio per ogni colonna, si applica alle
    riserve di testa della diagonale corrente per proiettare i pagamenti futuri,
    e si deriva la riserva totale.
    """
    n = triangle_paid.shape[0]
    if triangle_case.shape != triangle_paid.shape:
        raise ValueError("I due triangoli devono avere la stessa dimensione.")

    paid = triangle_paid.copy().astype(float)
    case = triangle_case.copy().astype(float)

    # Triangolo incrementale dei pagamenti
    incr = np.full_like(paid, np.nan)
    incr[:, 0] = paid[:, 0]  # colonna 0: pagato cumulato = incrementale
    for col in range(1, n):
        for row in range(n):
            if not np.isnan(paid[row, col]) and not np.isnan(paid[row, col - 1]):
                incr[row, col] = paid[row, col] - paid[row, col - 1]

    # Matrice dei ratio: incr[i, col] / case[i, col-1]
    ratio_matrix = np.full((n, n - 1), np.nan)
    for col in range(1, n):
        for row in range(n - col):  # solo celle storiche
            if (not np.isnan(incr[row, col]) and
                    not np.isnan(case[row, col - 1]) and
                    case[row, col - 1] > 0):
                ratio_matrix[row, col - 1] = incr[row, col] / case[row, col - 1]

    # Seleziona un ratio medio per colonna
    selected_ratios = []
    for col in range(n - 1):
        vals = ratio_matrix[~np.isnan(ratio_matrix[:, col]), col]
        if len(vals) == 0:
            selected_ratios.append(0.0)
            continue
        if tipo_media == "semplice":
            selected_ratios.append(float(np.mean(vals)))
        elif tipo_media == "mediana":
            selected_ratios.append(float(np.median(vals)))
        elif tipo_media == "ultimi3":
            selected_ratios.append(float(np.mean(vals[-3:])))
        else:  # volume: usa media semplice (non volume-weighted per i ratio)
            selected_ratios.append(float(np.mean(vals)))

    # Diagonale delle riserve di testa correnti
    case_diag = np.array([
        case[i, n - 1 - i] if not np.isnan(case[i, n - 1 - i]) else 0.0
        for i in range(n)
    ])

    # Diagonale dei pagati correnti
    paid_diag = _diagonale_attuale(triangle_paid)

    # Proietta i pagamenti futuri: per ogni anno immaturo, applica i ratios ai
    # case outstanding correnti in cascata (la riserva si riduce man mano che pago)
    riserve = np.zeros(n)
    for row in range(n):
        maturita_corrente = n - 1 - row  # indice colonna dell'ultima osservazione
        case_corrente = case_diag[row]
        pagato_futuro = 0.0
        for col in range(maturita_corrente + 1, n):
            idx_ratio = col - 1  # ratio[col] = incr[col] / case[col-1]
            if idx_ratio < len(selected_ratios):
                pagamento_stimato = selected_ratios[idx_ratio] * case_corrente
                pagato_futuro += pagamento_stimato
                # Aggiorna case outstanding residua (approssimazione)
                case_corrente = max(0.0, case_corrente - pagamento_stimato)
        riserve[row] = max(0.0, pagato_futuro)

    # Per l'anno più maturo (row=0) la riserva deve essere zero
    riserve[0] = 0.0

    ultimati = paid_diag + riserve

    return {
        "metodo": "Case Outstanding Development",
        "ratio_matrix": ratio_matrix,
        "selected_ratios": np.array(selected_ratios),
        "case_diagonale": case_diag,
        "paid_diagonale": paid_diag,
        "ultimati": ultimati,
        "pagati_attuali": paid_diag,
        "riserve_per_anno": riserve,
        "riserva_totale": float(riserve.sum()),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  6. BACK-TEST RETROSPETTIVO
# ══════════════════════════════════════════════════════════════════════════════

def _remove_diagonals(triangle: np.ndarray, k: int) -> np.ndarray:
    """
    Rimuove le ultime k diagonali dal triangolo (mette NaN).
    Restituisce il triangolo ridotto e il numero effettivo di anni rimossi.
    """
    n = triangle.shape[0]
    tri = triangle.copy().astype(float)
    # Le ultime k diagonali = le k righe più recenti nell'ultima colonna disponibile
    # In pratica: per ogni diagonale d (da 0 = principale a k-1), azzera
    # tutti gli elementi triangle[i,j] dove i+j == n-1-d per d in range(k)
    for d in range(k):
        for i in range(n):
            j = n - 1 - d - i
            if 0 <= j < n:
                tri[i, j] = np.nan
    return tri


def backtest(
    triangle: np.ndarray,
    n_diagonali: int = 1,
    premi: Optional[np.ndarray] = None,
    loss_ratio: float = 0.70,
    tipo_media: str = "volume",
) -> dict:
    """
    Back-test retrospettivo: rimuove le ultime n_diagonali dal triangolo,
    ricalcola tutti i metodi applicabili, e confronta gli ultimati proiettati
    con quelli osservati nella diagonale rimossa.

    Restituisce per ogni metodo e per ogni anno:
    - ultimato proiettato
    - ultimato vero (dalla diagonale rimossa)
    - errore assoluto e percentuale
    """
    n = triangle.shape[0]
    n_diagonali = min(n_diagonali, n - 2)  # deve restare almeno un triangolo 2×2

    tri_ridotto = _remove_diagonals(triangle, n_diagonali)

    # Diagonale "vera" (quella che vogliamo predire)
    # = ultima diagonale del triangolo originale non rimosso
    # Per semplicità: confrontiamo l'ultimato del triangolo originale (chain ladder full)
    # vs l'ultimato proiettato dal triangolo ridotto.
    # "Ultimato vero" = chain ladder sul triangolo completo (proxy).
    cl_full = chain_ladder(triangle, tipo_media=tipo_media)
    ultimati_veri = cl_full["ultimati"]

    risultati_bt = {}

    # Chain Ladder
    try:
        cl_rid = chain_ladder(tri_ridotto, tipo_media=tipo_media)
        risultati_bt["Chain Ladder"] = {
            "ultimati_proiettati": cl_rid["ultimati"],
            "ultimati_veri": ultimati_veri,
            "errori_assoluti": ultimati_veri - cl_rid["ultimati"],
            "errori_pct": (ultimati_veri - cl_rid["ultimati"]) / np.where(ultimati_veri > 0, ultimati_veri, np.nan) * 100,
        }
    except Exception:
        pass

    # BF (solo se premi disponibili)
    if premi is not None:
        try:
            bf_rid = bornhuetter_ferguson(tri_ridotto, premi, loss_ratio, tipo_media=tipo_media)
            risultati_bt["Bornhuetter-Ferguson"] = {
                "ultimati_proiettati": bf_rid["ultimati"],
                "ultimati_veri": ultimati_veri,
                "errori_assoluti": ultimati_veri - bf_rid["ultimati"],
                "errori_pct": (ultimati_veri - bf_rid["ultimati"]) / np.where(ultimati_veri > 0, ultimati_veri, np.nan) * 100,
            }
        except Exception:
            pass

        try:
            cc_rid = cape_cod(tri_ridotto, premi, tipo_media=tipo_media)
            risultati_bt["Cape Cod"] = {
                "ultimati_proiettati": cc_rid["ultimati"],
                "ultimati_veri": ultimati_veri,
                "errori_assoluti": ultimati_veri - cc_rid["ultimati"],
                "errori_pct": (ultimati_veri - cc_rid["ultimati"]) / np.where(ultimati_veri > 0, ultimati_veri, np.nan) * 100,
            }
        except Exception:
            pass

    return {
        "n_diagonali_rimosse": n_diagonali,
        "triangolo_ridotto": tri_ridotto,
        "ultimati_veri": ultimati_veri,
        "metodi": risultati_bt,
    }



# ══════════════════════════════════════════════════════════════════════════════
#  7. FREQUENCY-SEVERITY
# ══════════════════════════════════════════════════════════════════════════════

def frequency_severity(
    triangle_paid: np.ndarray,
    triangle_counts: np.ndarray,
    tipo_media: str = "volume",
    tail_factor_paid: float = 1.0,
    tail_factor_counts: float = 1.0,
) -> dict:
    """
    Frequency-Severity Technique (Friedland Cap. 11 - Approccio 1).
    
    Proietta separatamente:
    - Conteggi ultimati (chain ladder su triangle_counts)
    - Severity ultimata (chain ladder su triangolo severity = paid/counts)
    Ultimato F-S = conteggi ultimati × severity ultimata
    """
    n = triangle_paid.shape[0]
    if triangle_counts.shape != triangle_paid.shape:
        raise ValueError("I due triangoli devono avere la stessa dimensione.")

    # Triangolo severity: paid / counts (solo celle valide)
    triangle_severity = np.full_like(triangle_paid, np.nan)
    for i in range(n):
        for j in range(n):
            p = triangle_paid[i, j]
            c = triangle_counts[i, j]
            if not np.isnan(p) and not np.isnan(c) and c > 0:
                triangle_severity[i, j] = p / c

    # Chain Ladder separato su conteggi e severity
    cl_counts = chain_ladder(triangle_counts, tipo_media=tipo_media,
                              tail_factor=tail_factor_counts)
    cl_severity = chain_ladder(triangle_severity, tipo_media=tipo_media,
                                tail_factor=tail_factor_paid)

    ultimati_counts = cl_counts["ultimati"]
    ultimati_severity = cl_severity["ultimati"]
    ultimati_fs = ultimati_counts * ultimati_severity

    paid_diag = _diagonale_attuale(triangle_paid)
    riserve = np.maximum(ultimati_fs - paid_diag, 0.0)

    # Diagonale attuale severity (per confronto)
    severity_diag = np.array([
        triangle_severity[i, n - 1 - i]
        if not np.isnan(triangle_severity[i, n - 1 - i]) else 0.0
        for i in range(n)
    ])
    counts_diag = _diagonale_attuale(triangle_counts)

    return {
        "metodo": "Frequency-Severity",
        "triangle_severity": triangle_severity,
        "cl_counts": cl_counts,
        "cl_severity": cl_severity,
        "counts_diagonale": counts_diag,
        "severity_diagonale": severity_diag,
        "ultimati_counts": ultimati_counts,
        "ultimati_severity": ultimati_severity,
        "ultimati": ultimati_fs,
        "pagati_attuali": paid_diag,
        "riserve_per_anno": riserve,
        "riserva_totale": float(riserve.sum()),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: tabella riepilogativa riserve
# ══════════════════════════════════════════════════════════════════════════════

def tabella_riepilogo_riserve(anni: list, risultati: list[dict]) -> pd.DataFrame:
    """Crea una tabella comparativa tra più metodi di riserva."""
    df = pd.DataFrame({"Anno Accadimento": anni})
    for res in risultati:
        metodo = res["metodo"]
        df[f"Riserva {metodo} (€)"] = res["riserve_per_anno"]
        if "ultimati" in res:
            df[f"Ultimato {metodo} (€)"] = res["ultimati"]
    totale_row = {"Anno Accadimento": "TOTALE"}
    for col in df.columns[1:]:
        totale_row[col] = df[col].sum()
    df = pd.concat([df, pd.DataFrame([totale_row])], ignore_index=True)
    return df
