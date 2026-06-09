# Few-Shot Image Segmentation with Visual Prompt Adapter

Ten projekt implementuje system segmentacji obrazów typu "Few-Shot" przy użyciu modeli fundamentowych (DINOv2, CLIP) oraz lekkiego adaptera MLP współpracującego z dekoderem MobileSAM. Proces został podzielony na 4 niezależne etapy: od przygotowania danych po końcową ewaluację.

## Struktura Projektu
- `split_data.py` - Skrypt dzielący pełny zbiór danych na Train/Test.
- `precompute_features.py` - Skrypt do jednorazowej ekstrakcji cech (cache'owania).
- `train_sam.py` - Skrypt treningowy dla adapterów modeli fundamentowych.
- `eval.py` - Skrypt ewaluacyjny z analizą błędów.
- `fewshot_train/` & `fewshot_test/` - Zbiór danych FSS-1000 podzielony na dane treningowe (760 klas) i testowe (240 klas).
- `models/` - Folder docelowy przechowujący wyuczone wagi adapterów (`.pth`).
- `precomputed_train_.../` & `precomputed_test_.../` - Cache'owane wektory cech z modeli DINO/CLIP.
- `results_vis_.../` - Wygenerowane wizualizacje wyników segmentacji.

## Wymagania Środowiskowe
Aby poprawnie uruchomić projekt, upewnij się, że posiadasz środowisko obsługujące sprzętową akcelerację:
- Karta graficzna z obsługą CUDA (np. seria NVIDIA RTX 4060).
- PyTorch z zainstalowanym wsparciem CUDA (weryfikacja: `torch.cuda.is_available()`).
- Biblioteki Python: `transformers`, `opencv-python`, `numpy`, `tqdm`, `matplotlib`.
- Pobrane wagi modelu dekodera `mobile_sam.pt` w głównym katalogu.

## Instrukcja Uruchomienia (Pipeline)

Uruchamiaj poniższe skrypty po kolei w swoim środowisku wirtualnym:

### Krok 1: Podział zbioru danych
```bash
python split_data.py
```
*Dzieli surowe dane z folderu FSS-1000 na zbiór treningowy i testowy w oparciu o oficjalną listę 240 klas testowych, gwarantując brak przecieków danych (data leakage).*

### Krok 2: Ekstrakcja cech (Precomputing)
```bash
python precompute_features.py
```
*Przepuszcza obrazy przez modele fundamentowe (DINO-Small, DINO-Base, CLIP) oraz MobileSAM. Wyniki są zapisywane na dysku (`.pt`). Operacja jest obciążająca dla GPU, ale wykonuje się ją tylko raz.*

### Krok 3: Trening modeli
```bash
python train_sam.py
```
*Trenuje architekturę `VisualPromptAdapter` dla każdego z trzech wariantów przy użyciu wcześniej obliczonych cech. Wagi zapisywane są co określoną liczbę epok bezpośrednio do folderu `models/`.*

### Krok 4: Ewaluacja i analiza wyników
```bash
python eval.py
```
