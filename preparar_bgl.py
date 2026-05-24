"""
preparar_experiment_bgl_sense_tocar_model_train.py

Crea un experiment BGL compatible amb el teu pipeline actual SENSE modificar:
- model.py
- train.py
- data_loader.py

El que fa:
1) Llegeix data/raw/BGL.log del teu projecte principal.
2) Converteix BGL a un format compatible amb el teu data_loader actual:
   - data/processed/HDFS.log_structured.csv
   - data/raw/anomaly_label.csv
3) Crea una carpeta aïllada experiments_bgl/
4) Copia train.py, model.py i data_loader.py sense canviar-los.
5) Així pots entrenar BGL executant el mateix train.py, però dins d'experiments_bgl.

Ús:
    python preparar_experiment_bgl_sense_tocar_model_train.py

Després:
    cd experiments_bgl
    python train.py
"""

import csv
import os
import re
import shutil
from pathlib import Path


# ================= CONFIGURACIÓ =================
RAW_BGL_FILE = Path("data/raw/BGL.log")
EXPERIMENT_DIR = Path("experiments_bgl")

# Fitxers originals que NO es modificaran, només es copiaran iguals.
FILES_TO_COPY = ["model.py", "train.py", "data_loader.py"]

# BGL no té BlockId com HDFS. Per això creem "pseudo-BlockId" per finestres.
WINDOW_SIZE = 50      # nombre de logs per finestra
STRIDE = 50           # 50 = finestres sense solapament. Pots posar 25 si vols més dades.
MAX_LINES = None      # posa un número, ex. 1_000_000, si vols limitar per proves ràpides

# Sortides compatibles amb el train.py actual, dins d'experiments_bgl/
OUTPUT_STRUCTURED_NAME = "HDFS.log_structured.csv"
OUTPUT_LABEL_NAME = "anomaly_label.csv"
# =================================================


def ensure_original_files_exist():
    missing = [f for f in FILES_TO_COPY if not Path(f).exists()]
    if missing:
        raise FileNotFoundError(
            "No trobo aquests fitxers al directori actual: "
            + ", ".join(missing)
            + "\nExecuta aquest script des de la carpeta principal del projecte."
        )

    if not RAW_BGL_FILE.exists():
        raise FileNotFoundError(
            f"No trobo {RAW_BGL_FILE}.\n"
            "Posa el fitxer BGL.log a data/raw/BGL.log abans d'executar aquest script."
        )


def detect_bgl_label_and_message(line: str):
    """
    Detecta si una línia BGL és normal o anòmala i retorna el text útil del log.

    En el dataset BGL de Loghub, normalment el primer camp és:
      '-'  => Normal
      qualsevol altra etiqueta => Anomaly

    Exemple habitual:
      - 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 ...
      KERNEL 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 ...

    Si el fitxer no porta etiqueta al primer camp, es fa una detecció de seguretat
    basada en paraules de severitat.
    """
    parts = line.strip().split()
    if not parts:
        return 0, ""

    first = parts[0]

    # Cas habitual BGL etiquetat
    if first == "-":
        is_anomaly = 0
        payload = parts[1:]
    elif not re.fullmatch(r"\d+", first):
        # Si el primer camp no és un timestamp numèric i no és '-', assumim que és etiqueta d'anomalia
        is_anomaly = 1
        payload = parts[1:]
    else:
        # Fallback per si el BGL no porta label explícit
        severity_words = {"ERROR", "FATAL", "FAIL", "FAILED", "FAILURE", "WARN", "WARNING", "SEVERE", "PANIC"}
        is_anomaly = 1 if any(w.upper() in severity_words for w in parts) else 0
        payload = parts

    # Payload esperat sense label:
    # Timestamp Date Node Time NodeRepeat Type Component Level Content...
    # Per conservar senyal útil, agafem Type + Component + Level + Content si existeix.
    if len(payload) >= 8:
        message = " ".join(payload[5:])  # NodeRepeat/Type/Component/Level/Content aprox.
    else:
        message = " ".join(payload)

    return is_anomaly, message


def clean_bgl_template(message: str):
    """
    Normalitza valors dinàmics perquè missatges equivalents comparteixin plantilla.
    """
    text = message

    # Nodes BGL: R02-M1-N0-C:J12-U11, etc.
    text = re.sub(r"R\d+-M\d+-N\d+-C:J\d+-U\d+", "<NODE>", text)
    text = re.sub(r"R\d+-M\d+-N\d+", "<NODE>", text)

    # IPs i ports
    text = re.sub(r"\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?", "<IP>", text)

    # Dates i hores
    text = re.sub(r"\d{4}[.-]\d{2}[.-]\d{2}(?:[-T]\d{2}\.\d{2}\.\d{2}(?:\.\d+)?)?", "<DATE>", text)
    text = re.sub(r"\d{2}:\d{2}:\d{2}(?:\.\d+)?", "<TIME>", text)

    # Hexadecimals i adreces de memòria
    text = re.sub(r"0x[0-9A-Fa-f]+", "<HEX>", text)

    # Números llargs o variables
    text = re.sub(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?![A-Za-z])", "<NUM>", text)

    # Espais
    text = re.sub(r"\s+", " ", text).strip()

    return text if text else "<EMPTY>"


def iter_bgl_events(raw_file: Path):
    """
    Itera el fitxer BGL i produeix tuples:
    (line_id, raw_line, event_template, is_anomaly)
    """
    templates_map = {}
    with raw_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line_id, line in enumerate(f, start=1):
            if MAX_LINES is not None and line_id > MAX_LINES:
                break

            line = line.strip()
            if not line:
                continue

            is_anomaly, message = detect_bgl_label_and_message(line)
            template = clean_bgl_template(message)

            if template not in templates_map:
                templates_map[template] = f"E{len(templates_map) + 1}"

            event_id = templates_map[template]
            yield line_id, line, event_id, template, is_anomaly

    print(f"📊 Plantilles/EventId únics detectats: {len(templates_map)}")


def create_experiment_folder():
    if EXPERIMENT_DIR.exists():
        print(f"⚠️ La carpeta {EXPERIMENT_DIR} ja existeix.")
        answer = input("Vols eliminar-la i crear-la de nou? (s/n): ").strip().lower()
        if answer != "s":
            print("Procés cancel·lat. No s'ha modificat res.")
            raise SystemExit(0)
        shutil.rmtree(EXPERIMENT_DIR)

    (EXPERIMENT_DIR / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (EXPERIMENT_DIR / "data" / "processed").mkdir(parents=True, exist_ok=True)

    for file_name in FILES_TO_COPY:
        shutil.copy2(file_name, EXPERIMENT_DIR / file_name)

    print(f"✅ S'han copiat {', '.join(FILES_TO_COPY)} dins de {EXPERIMENT_DIR}/ sense modificar-los.")


def build_bgl_as_hdfs_compatible():
    structured_path = EXPERIMENT_DIR / "data" / "processed" / OUTPUT_STRUCTURED_NAME
    label_path = EXPERIMENT_DIR / "data" / "raw" / OUTPUT_LABEL_NAME

    events = list(iter_bgl_events(RAW_BGL_FILE))
    if not events:
        raise RuntimeError("No s'ha llegit cap esdeveniment de BGL.log.")

    total_windows = 0
    normal_windows = 0
    anomaly_windows = 0

    with structured_path.open("w", newline="", encoding="utf-8") as f_struct, \
         label_path.open("w", newline="", encoding="utf-8") as f_label:

        struct_writer = csv.writer(f_struct)
        label_writer = csv.writer(f_label)

        # Mateixa estructura que espera el teu data_loader.py
        struct_writer.writerow(["LineId", "BlockId", "Content", "EventId", "EventTemplate"])
        label_writer.writerow(["BlockId", "Label"])

        output_line_id = 1

        # Finestres consecutives o lliscants
        for start in range(0, max(len(events) - WINDOW_SIZE + 1, 0), STRIDE):
            window = events[start:start + WINDOW_SIZE]
            if len(window) < WINDOW_SIZE:
                continue

            block_id = f"bgl_window_{total_windows + 1:08d}"
            has_anomaly = any(e[4] == 1 for e in window)
            label = "Anomaly" if has_anomaly else "Normal"

            label_writer.writerow([block_id, label])

            for _, raw_line, event_id, template, _ in window:
                struct_writer.writerow([output_line_id, block_id, raw_line, event_id, template])
                output_line_id += 1

            total_windows += 1
            if has_anomaly:
                anomaly_windows += 1
            else:
                normal_windows += 1

            if total_windows % 10000 == 0:
                print(f"   ... finestres creades: {total_windows}", end="\r")

    print("\n✅ Dataset BGL convertit a format compatible amb el teu train.py")
    print(f"📁 CSV estructurat: {structured_path}")
    print(f"📁 Etiquetes:       {label_path}")
    print(f"📦 Finestres totals: {total_windows}")
    print(f"✅ Finestres normals: {normal_windows}")
    print(f"🚨 Finestres anòmales: {anomaly_windows}")

    if anomaly_windows < 2 or normal_windows < 2:
        print("\n⚠️ ALERTA: hi ha molt poques finestres d'una classe.")
        print("El train_test_split amb stratify pot fallar.")
        print("Prova de canviar WINDOW_SIZE o STRIDE, o revisa que BGL.log tingui etiquetes.")


def write_readme():
    readme_path = EXPERIMENT_DIR / "README_COM_EXECUTAR_BGL.txt"
    text = f"""VALIDACIÓ EXTERNA AMB BGL SENSE TOCAR MODEL.PY NI TRAIN.PY

Aquesta carpeta és un experiment aïllat.

S'han copiat sense modificar:
- model.py
- train.py
- data_loader.py

BGL s'ha convertit al format que ja espera el teu pipeline:
- data/processed/HDFS.log_structured.csv
- data/raw/anomaly_label.csv

Això és intencionat: el train.py original està escrit per llegir aquests noms.
No s'ha canviat cap línia del train.py.

PASSOS:

1) Entra a la carpeta:
   cd experiments_bgl

2) Executa el teu train original:
   python train.py

3) Els resultats sortiran a:
   Resultados_HDFS_Prova23/

Com que aquest experiment és BGL però el train.py no s'ha canviat,
pots renombrar la carpeta de resultats quan acabi:

   Resultados_HDFS_Prova23  ->  Resultados_BGL_Validacio

CONFIGURACIÓ UTILITZADA:
- WINDOW_SIZE = {WINDOW_SIZE}
- STRIDE = {STRIDE}
- MAX_LINES = {MAX_LINES}

IMPORTANT:
BGL no té BlockId com HDFS. Per això s'han creat BlockId artificials:
bgl_window_00000001, bgl_window_00000002, etc.

Cada finestra és anòmala si conté almenys un log anòmal.
"""
    readme_path.write_text(text, encoding="utf-8")
    print(f"📝 README creat: {readme_path}")


def main():
    print("🚀 Preparant experiment BGL compatible amb el train.py original")
    ensure_original_files_exist()
    create_experiment_folder()
    build_bgl_as_hdfs_compatible()
    write_readme()

    print("\n🎉 Fet!")
    print("Ara executa:")
    print(f"   cd {EXPERIMENT_DIR}")
    print("   python train.py")
    print("\nNo s'ha modificat el teu model.py ni el teu train.py originals.")


if __name__ == "__main__":
    main()
