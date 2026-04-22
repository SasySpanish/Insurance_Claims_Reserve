# 🛡️ Insurance Calculator Suite

Piattaforma Streamlit per il calcolo dei risarcimenti assicurativi e della riserva sinistri (IBNR).

## Funzionalità

### Risarcimento
- **Sinistro Singolo**: franchigia assoluta/relativa, scoperto (con min/max), massimale, analisi sensitività
- **Portafoglio Sinistri**: input manuale, CSV upload, generatore demo; limite aggregato annuo; KPI + grafici

### Riserva Sinistri (IBNR)
| Metodo | Descrizione |
|---|---|
| Chain Ladder | Fattori età-età dal triangolo cumulato |
| Bornhuetter-Ferguson | CL + perdita a priori (LR × premi) |
| Cape Cod | BF con ELR stimato empiricamente |
| Average Cost per Claim | N° IBNR × costo medio (± inflazione) |

## Struttura

```
insurance_app/
├── Home.py                         # Landing page
├── pages/
│   ├── 1_Risarcimento_Singolo.py
│   ├── 2_Risarcimento_Collettivo.py
│   └── 3_Riserva_Sinistri.py
├── utils/
│   ├── __init__.py
│   ├── calcoli_risarcimento.py     # logica risarcimenti
│   └── riserva_sinistri.py         # metodi IBNR
├── .streamlit/
│   └── config.toml
└── requirements.txt
```

## Deploy locale

```bash
pip install -r requirements.txt
streamlit run Home.py
```

## Deploy su Streamlit Cloud

1. Fork/push questo repository su GitHub
2. Vai su [share.streamlit.io](https://share.streamlit.io)
3. **New app** → seleziona il repo → `Main file: Home.py`
4. Click **Deploy**

> Nessuna variabile d'ambiente richiesta.

## Rami supportati

| Ramo | Franchigia default | Scoperto | Massimale |
|---|---|---|---|
| RC Auto | Assoluta €500 | No | €6.07M |
| Property / Incendio | Relativa €250 | 10% | €500K |
| Infortuni | Assoluta 5% inv. | No | €100K |
