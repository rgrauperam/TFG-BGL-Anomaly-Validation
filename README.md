# TFG - BGL Anomaly Validation

Aquest repositori conté l'adaptació del sistema de detecció d'anomalies al dataset BGL. Aquesta prova s'ha utilitzat com a validació externa del model desenvolupat en el Treball de Fi de Grau.

L'objectiu és comprovar si la metodologia basada en Transformer, Attention Pooling i Mean Pooling es pot adaptar a una base de dades diferent de HDFS.

---

## Descripció general

A diferència de HDFS, el dataset BGL no disposa d'un identificador equivalent al `BlockId`. Per aquest motiu, les seqüències d'entrada s'han construït mitjançant finestres consecutives de logs.

Cada finestra es classifica com a:

- `Normal`, si tots els logs de la finestra són normals.
- `Anomaly`, si almenys un log de la finestra està etiquetat com a alerta.

El model utilitzat manté la mateixa arquitectura que en la implementació principal:

- Transformer Encoder
- Attention Pooling
- Mean Pooling
- Classificador binari

Aquesta validació permet comprovar si el sistema pot mantenir un bon rendiment en un conjunt de logs amb una estructura diferent.

---

## Dataset necessari

Per executar el projecte és necessari descarregar prèviament el dataset BGL.

El dataset original no s'inclou en aquest repositori perquè el fitxer de logs és massa gran. Per aquest motiu, cada usuari ha de descarregar-lo manualment des del repositori públic Loghub.

```text
https://github.com/logpai/loghub
```

Cal descarregar el dataset **BGL** i col·locar el fitxer dins de la carpeta `data/raw/`.

L'estructura correcta ha de ser:

```text
data/
├── raw/
│   └── BGL.log
│
└── processed/
```

És important que el fitxer tingui exactament aquest nom:

```text
data/raw/BGL.log
```

Si el fitxer `BGL.log` no es troba dins de `data/raw/`, el preprocessament no es podrà executar.

---

## Execució del projecte

Per executar correctament el projecte, és important seguir l'ordre següent:

1. Primer s'ha d'executar `preparar_bgl.py`.
2. Després s'ha d'executar `train.py`.

Aquest ordre és necessari perquè el model no pot entrenar-se directament amb el fitxer original `BGL.log`. Primer cal transformar els logs originals en un format estructurat compatible amb el model.

### 1. Preparar les dades

```bash
python preparar_bgl.py
```

Aquest script llegeix:

```text
data/raw/BGL.log
```

i genera automàticament els fitxers:

```text
data/processed/BGL.log_structured.csv
data/raw/anomaly_label.csv
```

El fitxer `BGL.log_structured.csv` conté les seqüències d'esdeveniments generades a partir de finestres consecutives.

El fitxer `anomaly_label.csv` conté l'etiqueta de cada finestra, indicant si és normal o anòmala.

### 2. Entrenar el model

Un cop generats els fitxers processats, es pot executar:

```bash
python train.py
```

Aquest script utilitza:

```text
data/processed/BGL.log_structured.csv
data/raw/anomaly_label.csv
```

Durant l'entrenament es divideixen les dades en entrenament, validació i test. També es genera una carpeta de resultats amb el model entrenat, les gràfiques i la matriu de confusió.

L'ordre resumit és:

```bash
python preparar_bgl.py
python train.py
```

---

## Estructura del repositori

L'estructura principal del projecte és la següent:

```text
TFG-BGL-Anomaly-Validation/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── data_loader.py
├── model.py
├── preparar_bgl.py
├── train.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Fitxers principals

### `preparar_bgl.py`

Processa el fitxer original `BGL.log` i genera les seqüències mitjançant finestres consecutives de logs.

Com que BGL no té `BlockId`, el script crea identificadors artificials per a cada finestra, com per exemple:

```text
bgl_window_00000001
bgl_window_00000002
```

### `data_loader.py`

Carrega el fitxer processat i les etiquetes generades per `preparar_bgl.py`.

### `model.py`

Defineix l'arquitectura Transformer utilitzada per classificar les seqüències com a normals o anòmales.

### `train.py`

Entrena, valida i avalua el model sobre el dataset BGL processat.

---

## Instal·lació de dependències

Les llibreries necessàries es troben al fitxer `requirements.txt`.

Per instal·lar-les:

```bash
pip install -r requirements.txt
```

El fitxer `requirements.txt` inclou les dependències principals del projecte:

```text
torch
numpy
pandas
scikit-learn
matplotlib
seaborn
```

---

## Fitxers no inclosos al repositori

Alguns fitxers no s'inclouen al repositori perquè són massa grans o perquè es generen automàticament.

El fitxer `.gitignore` evita pujar:

```text
.vscode/
__pycache__/
*.pyc

data/raw/BGL.log
data/processed/

*.pt
*.pth
*.pkl
```

Per tant, cada usuari ha de descarregar el dataset original i executar el preprocessament.

---

## Autor

Roger Graupera Muñoz

Treball de Fi de Grau - Grau en Enginyeria Informàtica de Gestió i Sistemes d'Informació
