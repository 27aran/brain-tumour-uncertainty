import os
import imagehash
from PIL import Image
import kagglehub

print("START", flush=True)

PROJECT_DIR = None
LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']
THRESHOLD = 5  # Hamming-Distanz - je kleiner, desto strenger (0 = identisch)

def collect_files():
    """Sammelt alle Bildpfade mit Info, aus welchem Split sie stammen."""
    files = []
    for data_type in ['Training', 'Testing']:
        for label in LABELS:
            folder_path = os.path.join(PROJECT_DIR, data_type, label)
            for root, dirs, fs in os.walk(folder_path):
                for f in fs:
                    if f.endswith(".jpg"):
                        files.append(os.path.join(root, f))
    return files

def compute_hashes(files):
    """Berechnet Perceptual Hash für jede Datei."""
    hashes = {}
    for i, file_path in enumerate(files):
        try:
            img = Image.open(file_path)
            h = imagehash.phash(img)
            hashes[file_path] = h
        except Exception as e:
            print(f"Fehler bei {file_path}: {e}", flush=True)
        if i % 500 == 0:
            print(f"{i}/{len(files)} Bilder gehasht...", flush=True)
    return hashes

def find_and_remove_duplicates(hashes):
    """Vergleicht alle Hashes paarweise, entfernt Duplikate (Training hat Vorrang)."""
    items = list(hashes.items())
    to_remove = set()
    duplicate_count = 0
    total_comparisons = len(items)

    for i in range(len(items)):
        path_i, hash_i = items[i]
        if path_i in to_remove:
            continue
        for j in range(i + 1, len(items)):
            path_j, hash_j = items[j]
            if path_j in to_remove:
                continue
            distance = hash_i - hash_j
            if distance <= THRESHOLD:
                is_i_training = f"{os.sep}Training{os.sep}" in path_i
                is_j_training = f"{os.sep}Training{os.sep}" in path_j

                if is_i_training and not is_j_training:
                    to_remove.add(path_j)
                elif is_j_training and not is_i_training:
                    to_remove.add(path_i)
                    break
                else:
                    to_remove.add(path_j)

                duplicate_count += 1

        if i % 500 == 0:
            print(f"Vergleiche: {i}/{total_comparisons}", flush=True)

    print(f"Gefundene Duplikate: {duplicate_count}", flush=True)
    print(f"Zu löschende Dateien: {len(to_remove)}", flush=True)

    for path in to_remove:
        print(f"Removing: {path}", flush=True)
        os.remove(path)

    print(f"Fertig. {len(to_remove)} Dateien gelöscht.", flush=True)

if __name__ == '__main__':
    print("Vor Download-Check...", flush=True)
    PROJECT_DIR = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
    print("Nach Download-Check...", flush=True)

    print("Sammle Dateien...", flush=True)
    files = collect_files()
    print(f"{len(files)} Dateien gefunden.", flush=True)

    print("Berechne Perceptual Hashes...", flush=True)
    hashes = compute_hashes(files)

    print("Suche und entferne Duplikate...", flush=True)
    find_and_remove_duplicates(hashes)