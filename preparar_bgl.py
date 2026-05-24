import csv
import re
from pathlib import Path

RAW_BGL_FILE = Path("data/raw/BGL.log")

OUTPUT_STRUCTURED_FILE = Path("data/processed/BGL.log_structured.csv")
OUTPUT_LABEL_FILE = Path("data/raw/anomaly_label.csv")

WINDOW_SIZE = 50
STRIDE = 50
MAX_LINES = None


def ensure_input_exists():
    if not RAW_BGL_FILE.exists():
        raise FileNotFoundError(
            f"No trobo {RAW_BGL_FILE}.\n"
            "Has de col·locar el fitxer BGL.log dins de data/raw/"
        )


def detect_bgl_label_and_message(line: str):
    """
    Detecta si una línia BGL és normal o anòmala.

    En el dataset BGL de Loghub, normalment el primer camp és:
      '-'  => Normal
      qualsevol altra etiqueta => Anomaly
    """
    parts = line.strip().split()

    if not parts:
        return 0, ""

    first = parts[0]

    if first == "-":
        is_anomaly = 0
        payload = parts[1:]
    elif not re.fullmatch(r"\d+", first):
        is_anomaly = 1
        payload = parts[1:]
    else:
        # Fallback per si el fitxer no porta etiqueta explícita
        severity_words = {
            "ERROR", "FATAL", "FAIL", "FAILED", "FAILURE",
            "WARN", "WARNING", "SEVERE", "PANIC"
        }
        is_anomaly = 1 if any(w.upper() in severity_words for w in parts) else 0
        payload = parts

    # Payload aproximat:
    # Timestamp Date Node Time NodeRepeat Type Component Level Content...
    if len(payload) >= 8:
        message = " ".join(payload[5:])
    else:
        message = " ".join(payload)

    return is_anomaly, message


def clean_bgl_template(message: str):
    """
    Normalitza valors dinàmics perquè missatges equivalents comparteixin plantilla.
    """
    text = message

    # Nodes BGL
    text = re.sub(r"R\d+-M\d+-N\d+-C:J\d+-U\d+", "<NODE>", text)
    text = re.sub(r"R\d+-M\d+-N\d+", "<NODE>", text)

    # IPs i ports
    text = re.sub(r"\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?", "<IP>", text)

    # Dates i hores
    text = re.sub(
        r"\d{4}[.-]\d{2}[.-]\d{2}(?:[-T]\d{2}\.\d{2}\.\d{2}(?:\.\d+)?)?",
        "<DATE>",
        text
    )
    text = re.sub(r"\d{2}:\d{2}:\d{2}(?:\.\d+)?", "<TIME>", text)

    # Hexadecimals
    text = re.sub(r"0x[0-9A-Fa-f]+", "<HEX>", text)

    # Números
    text = re.sub(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?![A-Za-z])", "<NUM>", text)

    # Espais
    text = re.sub(r"\s+", " ", text).strip()

    return text if text else "<EMPTY>"


def build_bgl_dataset():
    ensure_input_exists()

    OUTPUT_STRUCTURED_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_LABEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    templates_map = {}
    window_buffer = []

    total_lines = 0
    output_line_id = 1
    total_windows = 0
    normal_windows = 0
    anomaly_windows = 0

    with RAW_BGL_FILE.open("r", encoding="utf-8", errors="ignore") as f_in, \
         OUTPUT_STRUCTURED_FILE.open("w", newline="", encoding="utf-8") as f_struct, \
         OUTPUT_LABEL_FILE.open("w", newline="", encoding="utf-8") as f_label:

        struct_writer = csv.writer(f_struct)
        label_writer = csv.writer(f_label)

        struct_writer.writerow(["LineId", "BlockId", "Content", "EventId", "EventTemplate"])
        label_writer.writerow(["BlockId", "Label"])

        for line in f_in:
            total_lines += 1

            if MAX_LINES is not None and total_lines > MAX_LINES:
                break

            line = line.strip()
            if not line:
                continue

            is_anomaly, message = detect_bgl_label_and_message(line)
            template = clean_bgl_template(message)

            if template not in templates_map:
                templates_map[template] = f"E{len(templates_map) + 1}"

            event_id = templates_map[template]

            window_buffer.append((line, event_id, template, is_anomaly))

            if len(window_buffer) == WINDOW_SIZE:
                total_windows += 1
                block_id = f"bgl_window_{total_windows:08d}"

                has_anomaly = any(event[3] == 1 for event in window_buffer)
                label = "Anomaly" if has_anomaly else "Normal"

                label_writer.writerow([block_id, label])

                for raw_line, event_id, template, _ in window_buffer:
                    struct_writer.writerow(
                        [output_line_id, block_id, raw_line, event_id, template]
                    )
                    output_line_id += 1

                if has_anomaly:
                    anomaly_windows += 1
                else:
                    normal_windows += 1

                window_buffer = window_buffer[STRIDE:]

                if total_windows % 10000 == 0:
                    print(f"   ... finestres creades: {total_windows}", end="\r")

    print("\nDataset BGL processat correctament")
    print(f"CSV estructurat: {OUTPUT_STRUCTURED_FILE}")
    print(f"Etiquetes:       {OUTPUT_LABEL_FILE}")
    print(f"Plantilles/EventId únics: {len(templates_map)}")
    print(f"Finestres totals: {total_windows}")
    print(f"Finestres normals: {normal_windows}")
    print(f"Finestres anòmales: {anomaly_windows}")

    if anomaly_windows < 2 or normal_windows < 2:
        print("\nALERTA: hi ha molt poques finestres d'una classe.")
        print("El train_test_split amb stratify pot fallar.")
        print("Prova de canviar WINDOW_SIZE o STRIDE, o revisa que BGL.log tingui etiquetes.")


def main():
    print("Preparant dataset BGL")
    print(f"Fitxer d'entrada: {RAW_BGL_FILE}")
    print(f"WINDOW_SIZE: {WINDOW_SIZE}")
    print(f"STRIDE: {STRIDE}")

    build_bgl_dataset()

    print("\nFet!")
    print("Ara pots executar:")
    print("   python train.py")


if __name__ == "__main__":
    main()