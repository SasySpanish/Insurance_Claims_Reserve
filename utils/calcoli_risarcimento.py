"""
Modulo calcolo risarcimenti assicurativi.
Gestisce: franchigia relativa/assoluta, scoperto, massimale, singoli e collettivi.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class TipoFranchigia(Enum):
    NESSUNA = "Nessuna"
    ASSOLUTA = "Assoluta"
    RELATIVA = "Relativa"


class RamoAssicurativo(Enum):
    RC_AUTO = "RC Auto"
    PROPERTY = "Property / Incendio"
    INFORTUNI = "Infortuni"


@dataclass
class ConfigPolizza:
    """Parametri della polizza assicurativa."""
    ramo: str
    tipo_franchigia: str
    franchigia: float = 0.0
    massimale: Optional[float] = None
    scoperto_perc: float = 0.0          # % a carico dell'assicurato dopo franchigia
    scoperto_minimo: float = 0.0        # scoperto minimo in €
    scoperto_massimo: Optional[float] = None   # tetto scoperto in €
    limite_sinistro: Optional[float] = None    # per sinistro (RC Auto)
    limite_aggregato: Optional[float] = None   # annuo aggregato


@dataclass
class Sinistro:
    """Singolo sinistro."""
    id: str
    danno_lordo: float
    anno_accadimento: int
    anno_denuncia: int
    liquidato: bool = False
    note: str = ""


def calcola_risarcimento_singolo(danno: float, cfg: ConfigPolizza) -> dict:
    """
    Calcola il risarcimento netto per un singolo sinistro.
    Logica:
      1. Applica franchigia (assoluta o relativa)
      2. Applica scoperto sulla parte eccedente
      3. Applica massimale / limite per sinistro
    Restituisce breakdown dettagliato.
    """
    result = {
        "danno_lordo": danno,
        "franchigia_applicata": 0.0,
        "base_dopo_franchigia": 0.0,
        "scoperto_importo": 0.0,
        "risarcimento_lordo": 0.0,
        "risarcimento_netto": 0.0,
        "note": []
    }

    # ── FRANCHIGIA ─────────────────────────────────────────────────────────────
    if cfg.tipo_franchigia == TipoFranchigia.ASSOLUTA.value:
        # La franchigia si deduce sempre (danno rimane a carico assicurato fino a soglia)
        if danno <= cfg.franchigia:
            result["franchigia_applicata"] = danno
            result["note"].append(f"Danno ≤ franchigia assoluta ({cfg.franchigia:,.2f}€): nessun risarcimento.")
            return result
        base = danno - cfg.franchigia
        result["franchigia_applicata"] = cfg.franchigia
        result["note"].append(f"Franchigia assoluta detratta: {cfg.franchigia:,.2f}€")

    elif cfg.tipo_franchigia == TipoFranchigia.RELATIVA.value:
        # Sotto soglia → tutto a carico assicurato; sopra soglia → pagato interamente
        if danno < cfg.franchigia:
            result["franchigia_applicata"] = danno
            result["note"].append(f"Danno < franchigia relativa ({cfg.franchigia:,.2f}€): nessun risarcimento.")
            return result
        base = danno
        result["franchigia_applicata"] = 0.0
        result["note"].append(f"Danno ≥ franchigia relativa: risarcimento sull'intero danno.")

    else:  # NESSUNA
        base = danno

    result["base_dopo_franchigia"] = base

    # ── SCOPERTO ───────────────────────────────────────────────────────────────
    scoperto = 0.0
    if cfg.scoperto_perc > 0:
        scoperto = base * cfg.scoperto_perc / 100
        if cfg.scoperto_minimo > 0:
            scoperto = max(scoperto, cfg.scoperto_minimo)
        if cfg.scoperto_massimo is not None:
            scoperto = min(scoperto, cfg.scoperto_massimo)
        result["note"].append(
            f"Scoperto {cfg.scoperto_perc}%: {scoperto:,.2f}€"
            + (f" (min {cfg.scoperto_minimo:,.2f}€)" if cfg.scoperto_minimo else "")
            + (f" (max {cfg.scoperto_massimo:,.2f}€)" if cfg.scoperto_massimo else "")
        )

    result["scoperto_importo"] = scoperto
    risarcimento = base - scoperto
    result["risarcimento_lordo"] = risarcimento

    # ── MASSIMALE / LIMITE PER SINISTRO ────────────────────────────────────────
    limite = None
    if cfg.limite_sinistro is not None:
        limite = cfg.limite_sinistro
        result["note"].append(f"Limite per sinistro applicato: {limite:,.2f}€")
    elif cfg.massimale is not None:
        limite = cfg.massimale
        result["note"].append(f"Massimale applicato: {limite:,.2f}€")

    if limite is not None and risarcimento > limite:
        risarcimento = limite

    result["risarcimento_netto"] = max(risarcimento, 0.0)
    return result


def calcola_risarcimento_collettivo(sinistri: List[dict], cfg: ConfigPolizza) -> pd.DataFrame:
    """
    Calcola risarcimento per un portafoglio di sinistri,
    applicando eventuale limite aggregato annuo.
    """
    rows = []
    totale_liquidato = 0.0

    for s in sinistri:
        danno = s["danno_lordo"]
        res = calcola_risarcimento_singolo(danno, cfg)

        # Limite aggregato annuo
        if cfg.limite_aggregato is not None:
            residuo = max(cfg.limite_aggregato - totale_liquidato, 0.0)
            if res["risarcimento_netto"] > residuo:
                res["risarcimento_netto"] = residuo
                res["note"].append(f"Limite aggregato raggiunto: capped a {residuo:,.2f}€")
        totale_liquidato += res["risarcimento_netto"]

        rows.append({
            "ID Sinistro": s.get("id", "—"),
            "Anno Acc.": s.get("anno_accadimento", "—"),
            "Anno Den.": s.get("anno_denuncia", "—"),
            "Danno Lordo (€)": danno,
            "Franchigia (€)": res["franchigia_applicata"],
            "Base (€)": res["base_dopo_franchigia"],
            "Scoperto (€)": res["scoperto_importo"],
            "Risarcimento Netto (€)": res["risarcimento_netto"],
            "Note": "; ".join(res["note"]) if res["note"] else "—",
        })

    return pd.DataFrame(rows)


# ── Parametri specifici per ramo ─────────────────────────────────────────────

RAMO_DEFAULTS = {
    RamoAssicurativo.RC_AUTO.value: {
        "franchigia_default": 500.0,
        "tipo_franchigia_default": TipoFranchigia.ASSOLUTA.value,
        "massimale_default": 6_070_000.0,   # minimo legale RCA Italia 2024
        "scoperto_default": 0.0,
        "descrizione": "RC Auto: limite minimo di legge 6.07M€ per danni a persone, 1.22M€ per cose.",
    },
    RamoAssicurativo.PROPERTY.value: {
        "franchigia_default": 250.0,
        "tipo_franchigia_default": TipoFranchigia.RELATIVA.value,
        "massimale_default": 500_000.0,
        "scoperto_default": 10.0,
        "descrizione": "Property/Incendio: tipicamente con scoperto 10-20% e franchigia relativa.",
    },
    RamoAssicurativo.INFORTUNI.value: {
        "franchigia_default": 5.0,   # % invalidità permanente
        "tipo_franchigia_default": TipoFranchigia.ASSOLUTA.value,
        "massimale_default": 100_000.0,
        "scoperto_default": 0.0,
        "descrizione": "Infortuni: franchigia in % di invalidità permanente; capitale assicurato come massimale.",
    },
}
