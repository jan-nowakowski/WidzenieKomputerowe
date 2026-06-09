import os
import shutil
from tqdm import tqdm

TEST_FILE = 'fss_test_set.txt'
SOURCE_DIR = './fewshot_data/'  # <--- Zmienione na nazwę Twojego folderu!
TRAIN_DIR = './fewshot_train/'
TEST_DIR = './fewshot_test/'

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# Wczytywanie klas testowych
with open(TEST_FILE, 'r') as f:
    test_classes = [line.strip() for line in f.readlines() if line.strip()]

all_classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]

print("Rozpoczynam podział pełnego zbioru danych...")
for cls in tqdm(all_classes):
    src_path = os.path.join(SOURCE_DIR, cls)

    # Przenosimy do test, jeśli klasa jest na liście, inaczej do train
    if cls in test_classes:
        shutil.move(src_path, os.path.join(TEST_DIR, cls))
    else:
        shutil.move(src_path, os.path.join(TRAIN_DIR, cls))

print("Gotowe! Dane przeniesione do 'fewshot_train' (760 klas) i 'fewshot_test' (240 klas).")