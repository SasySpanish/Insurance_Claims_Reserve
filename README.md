# 📐 Insurance Claims Reserve

**Applicazione Streamlit per il calcolo e la valutazione delle riserve sinistri IBNR**  
Metodologie attuariali standard · Diagnostica · Report scaricabile

---

## Indice

1. [Panoramica](#panoramica)
2. [Struttura del progetto](#struttura-del-progetto)
3. [Installazione](#installazione)
4. [Utilizzo](#utilizzo)
5. [Metodologie implementate](#metodologie-implementate)
6. [Modulo diagnostica](#modulo-diagnostica)
7. [Report di valutazione](#report-di-valutazione)
8. [Architettura del codice](#architettura-del-codice)
9. [Stack tecnologico](#stack-tecnologico)
10. [Roadmap](#roadmap)

---

## Panoramica

`insurance_claims_reserve` è un'applicazione web sviluppata con **Streamlit** per la stima delle riserve sinistri IBNR (*Incurred But Not Reported*) secondo le principali metodologie attuariali descritte in *Friedland – Estimating Unpaid Claims Using Basic Techniques* (CAS, 2010).

Il progetto nasce come strumento di portfolio per dimostrare la combinazione di background attuariale-statistico e competenze in data engineering / Python. Non è un software certificato per uso regolamentare.

**Funzionalità principali:**

- Inserimento e validazione del triangolo dei pagamenti cumulati
- Calcolo riserve IBNR con 6 metodologie indipendenti
- Selezione LDF con giudizio attuariale (outlier, esclusione anni, medie multiple)
- Diagnostica attuariale con grafici interattivi Plotly
- Back-test retrospettivo con RMSE per metodo
- Report di valutazione esportabile in HTML e PDF
- Interfaccia con gradiente teal-blu, ottimizzata per uso professionale

---

## Struttura del progetto

```
insurance_claims_reserve/
│
├── Riserva_Sinistri.py          # Applicazione Streamlit (entry point)
│
├── utils/
│   ├── __init__.py              # Esporta tutte le funzioni pubbliche
│   ├── riserva_sinistri.py      # Metodi attuariali core (backend)
│   ├── diagnostica.py           # Diagnostica, selezione LDF, report
│   └── calcoli_risarcimento.py  # Calcolo risarcimento singolo/collettivo
│
├── requirements.txt
└── README.md
```

---

## Installazione

### Prerequisiti

- Python 3.10+
- pip

### Setup locale

```bash
git clone https://github.com/tuo-username/insurance_claims_reserve.git
cd insurance_claims_reserve

pip install -r requirements.txt
streamlit run Riserva_Sinistri.py
```

### requirements.txt

```
streamlit
numpy
pandas
plotly
kaleido          # necessario per export PNG dei grafici nel report
weasyprint       # opzionale: per generazione PDF del report
```

> **Nota su weasyprint**: richiede librerie di sistema (`libpango`, `libcairo`).  
> Su Ubuntu: `sudo apt-get install libpango-1.0-0 libcairo2`.  
> Se non disponibile, il report HTML rimane pienamente funzionale.

---

## Utilizzo

### 1. Inserimento triangolo

Nella tab **📥 Triangolo** si inseriscono i pagamenti cumulati in formato n×n. Le celle della diagonale superiore destra (dati futuri) sono automaticamente disabilitate.

È disponibile un **dataset demo precaricato** per testare immediatamente tutte le funzionalità. Prima di procedere al calcolo, il sistema esegue validazione automatica che segnala:

- Valori negativi
- Righe non monotone (dato non cumulato)
- NaN sulla diagonale corrente

### 2. Calcolo riserve

Ogni metodo ha una tab dedicata con parametri configurabili. I risultati vengono salvati in sessione e resi disponibili nella tab **Evaluation** per il confronto.

### 3. Diagnostica e Report

Nella tab **🔬 Diagnostica** è possibile:

- Analizzare lo sviluppo cumulato per accident year
- Confrontare le medie LDF
- Inserire il triangolo degli incurred per il ratio Paid/Incurred
- Inserire i triangoli dei sinistri chiusi/denunciati per i closure rates
- Identificare anni e intervalli anomali
- Generare e scaricare il report di valutazione

---

## Metodologie implementate

### ⛓️ Chain Ladder (Capitolo 7 – Friedland)

Metodo di sviluppo standard. I pagamenti futuri vengono proiettati moltiplicando la diagonale attuale per i fattori età-età selezionati.

**Parametri configurabili:**
- Tipo di media: volume-weighted, semplice, mediana, ultimi 3 anni
- Override manuale di ogni fattore età-età
- Tail factor
- Esclusione manuale di accident years (es. 2020 per effetto Covid)
- Rimozione outlier (IQR o Z-score)

**Output:** triangolo completato, LDF selezionati, CDF cumulativi, riserve per anno.

---

### 📊 Bornhuetter-Ferguson (Capitolo 9 – Friedland)

Credibility blend tra chain ladder e perdita attesa a priori.

```
Ultimato_BF = Pagato + (1 - % sviluppato) × (Premio × ELR_apriori)
```

La credibilità assegnata all'esperienza reale cresce con la maturità dell'anno. Include diagnostica automatica che segnala la divergenza percentuale tra l'ELR selezionato a priori e l'ELR empirico Cape Cod: se superiore al 15%, viene mostrato un warning.

---

### 🎣 Cape Cod / Stanard-Buhlmann (Capitolo 10 – Friedland)

Identico al BF nella struttura, ma l'ELR non è selezionato a priori: viene stimato direttamente dai dati.

```
ELR = Σ Pagati / Σ (Premi × % sviluppato)
```

Elimina la soggettività nella scelta dell'ELR. Include warning esplicito sull'importanza di utilizzare premi corretti per il livello tariffario (*rate-level adjusted*).

---

### 🔢 Average Cost per Claim / ACPC (Capitolo 11 – Friedland)

```
Riserva = N° sinistri IBNR × Costo medio × (1 + inflazione)^t
```

Il numero di sinistri IBNR può essere inserito manualmente o **stimato automaticamente** tramite chain ladder applicato al triangolo dei conteggi cumulati.

---

### 📈 Frequency-Severity (Capitolo 11 – Friedland)

Proiezione separata di frequenza e severità, poi combinazione.

```
Ultimato_FS = Conteggi_ultimati × Severity_ultimata
```

Richiede un secondo triangolo con i conteggi cumulati dei sinistri. I due triangoli (conteggi e severity = paid/counts) vengono proiettati indipendentemente con chain ladder, permettendo di verificare se la divergenza rispetto agli altri metodi deriva da anomalie nella frequenza o nella severità.

---

### 🏛️ Case Outstanding Development – Wiser (Capitolo 12 – Friedland)

Analizza il rapporto tra pagamenti incrementali e riserve di testa aperte all'inizio del periodo.

```
ratio_{i,j} = (Paid_{i,j} - Paid_{i,j-1}) / CaseOS_{i,j-1}
```

I ratio medi per colonna vengono applicati alle riserve di testa correnti per stimare i pagamenti futuri. Adatto a linee *claims-made* o portafogli dove la quasi totalità dei sinistri è già nota.

---

### 🔍 Evaluation & Back-test (Capitolo 15 – Friedland)

Confronto multi-metodo e validazione retrospettiva.

**Confronto:** tutti i metodi eseguiti vengono visualizzati in parallelo con grouped bar chart per anno e totali. Se lo spread tra metodi supera il 20% del valore massimo, viene segnalato un alert con indicazione di analizzare le cause prima di selezionare la stima finale.

**Back-test:** rimozione delle ultime k diagonali (1–3, a scelta), ricalcolo dei metodi sul triangolo ridotto, confronto con gli ultimati del triangolo completo (proxy degli ultimati veri). Output: errore assoluto, errore percentuale e RMSE per metodo, con visualizzazione degli errori per anno e per metodo.

---

## Modulo diagnostica

Il modulo `utils/diagnostica.py` è indipendente dal backend dei metodi e può essere importato autonomamente.

### `select_ldf`

```python
from utils.diagnostica import select_ldf

sel = select_ldf(
    triangle,
    anni_label,
    method="weighted",        # weighted | all | last3 | last5 | trimmed
    outlier_method="iqr",     # iqr | zscore
    remove_outliers=True,
    exclude_years=[1],        # indici 0-based degli anni da escludere
    tail_factor=1.0,
)

print(sel.summary)      # DataFrame con tutte le medie per colonna
print(sel.selected)     # np.array LDF selezionati
print(sel.notes)        # lista di note diagnostiche testuali
```

Restituisce un oggetto `LDFSelection` con campi:

| Campo | Tipo | Descrizione |
|---|---|---|
| `selected` | `np.ndarray` | LDF selezionati |
| `summary` | `pd.DataFrame` | Tutte le medie per colonna |
| `outliers` | `dict` | Celle outlier per colonna |
| `excluded_years` | `list[int]` | Indici anni esclusi |
| `method` | `str` | Metodo utilizzato |
| `high_cv_cols` | `list[int]` | Colonne ad alta variabilità (CV > 10%) |
| `notes` | `list[str]` | Note diagnostiche testuali |

### `detect_anomalies`

```python
from utils.diagnostica import detect_anomalies

anom = detect_anomalies(triangle, anni_label, outlier_method="iqr")
# anom["outlier_cells"]   → lista celle LDF anomale
# anom["unstable_cols"]   → colonne con CV > 10%
# anom["anomalous_years"] → anni con ≥2 LDF outlier
# anom["summary"]         → DataFrame riepilogativo per anno
```

### Grafici diagnostici

Tutte le funzioni grafiche restituiscono `(figura, DataFrame_dati)`.

```python
from utils.diagnostica import (
    plot_development,
    plot_ldf_comparison,
    plot_paid_to_incurred,
    plot_closure_rates,
)

fig_dev, df_dev = plot_development(triangle, anni_label)
fig_ldf, df_ldf = plot_ldf_comparison(triangle, anni_label, ldf_selection=sel)
fig_pi,  df_pi  = plot_paid_to_incurred(triangle_paid, triangle_incurred, anni_label)
fig_cr,  df_cr  = plot_closure_rates(triangle_closed, triangle_reported, anni_label)
```

---

## Report di valutazione

```python
from utils.diagnostica import generate_evaluation_report

html_bytes, pdf_bytes = generate_evaluation_report(
    triangle=triangle,
    anni_label=anni,
    ldf_selection=sel,
    riserve_risultati=[res_cl, res_bf, res_cc],   # output dei metodi
    anomalie=anom,
    triangle_incurred=triangle_incurred,           # opzionale
    triangle_closed=triangle_closed,               # opzionale
    triangle_reported=triangle_reported,           # opzionale
    titolo="Report Riserve 2024",
    produrre_pdf=True,                             # richiede weasyprint
)

with open("report.html", "wb") as f:
    f.write(html_bytes)
```

Il report include: sintesi LDF selezionati, note diagnostiche, grafici di sviluppo, confronto medie LDF, Paid/Incurred, Closure Rate, anomalie rilevate, sintesi riserve per metodo. I grafici sono embedded come PNG base64. Lo stile è consulenziale con gradiente header teal-verde.

---

## Architettura del codice

```
utils/riserva_sinistri.py
│
├── validate_triangle()          → validazione input, lista warning
├── _diagonale_attuale()         → helper diagonale osservata
├── age_to_age_matrix()          → matrice link ratio n×(n-1)
├── compute_factors()            → fattori aggregati (4 tipi di media)
├── cdfs_from_factors()          → CDF cumulativi da fattori
│
├── chain_ladder()               → metodo 1
├── bornhuetter_ferguson()       → metodo 2
├── cape_cod()                   → metodo 3
├── average_cost_per_claim()     → metodo 4
├── stima_conteggi_da_triangolo()→ helper ACPC
├── frequency_severity()         → metodo 5
├── case_outstanding_development()→ metodo 6
│
├── backtest()                   → back-test retrospettivo
└── tabella_riepilogo_riserve()  → DataFrame comparativo

utils/diagnostica.py
│
├── LDFSelection                 → dataclass risultato selezione
├── select_ldf()                 → selezione LDF con giudizio attuariale
│
├── plot_development()           → sviluppo cumulato per AY
├── plot_ldf_comparison()        → confronto medie LDF
├── plot_paid_to_incurred()      → ratio paid/incurred
├── plot_closure_rates()         → tasso di chiusura
├── detect_anomalies()           → identificazione anomalie
│
└── generate_evaluation_report() → report HTML + PDF
```

### Principi di design

- **Nessuna funzione esistente modificata** nelle successive iterazioni: ogni aggiunta è additiva.
- **Output strutturati**: ogni metodo restituisce un dizionario con chiavi esplicite, non tuple posizionali.
- **Validazione non bloccante**: `validate_triangle` restituisce warning come lista di stringhe; è l'interfaccia a decidere come mostrarli.
- **Vettorizzazione NumPy**: i calcoli sui triangoli evitano loop Python dove possibile.
- **Separazione backend/UI**: tutta la logica matematica è in `utils/`, l'interfaccia Streamlit si limita a raccogliere input e visualizzare output.

---

## Stack tecnologico

| Libreria | Utilizzo |
|---|---|
| `streamlit` | Interfaccia web |
| `numpy` | Calcolo matriciale sui triangoli |
| `pandas` | Tabelle e output strutturati |
| `plotly` | Grafici interattivi e statici |
| `kaleido` | Export PNG per report embedded |
| `weasyprint` | Conversione HTML → PDF (opzionale) |

---

## Roadmap

- [ ] Integrazione RAG: upload PDF earnings call / relazioni tecniche con risposta LLM contestuale
- [ ] MLOps: containerizzazione Docker + deploy su cloud (Railway / Render)
- [ ] Berquist-Sherman: aggiustamento triangoli per cambiamenti operativi (Cap. 13 Friedland)
- [ ] Salvataggio sessione: export/import triangoli in formato JSON o Excel
- [ ] Test unitari su tutti i metodi con `pytest`
- [ ] FastAPI endpoint per utilizzo programmatico del backend

---

## Riferimenti

- Friedland, J. (2010). *Estimating Unpaid Claims Using Basic Techniques*. Casualty Actuarial Society.
- CAS Study Note on Development Techniques
- Wiser, R.F. (1994). *Loss Reserving*. In Foundations of Casualty Actuarial Science.

---

> Progetto sviluppato come parte di un portfolio data science / AI engineering.  
> Background: Laurea Magistrale in Statistica Attuariale · Esperienza in operazioni logistiche · Transizione verso ruoli data/AI.
