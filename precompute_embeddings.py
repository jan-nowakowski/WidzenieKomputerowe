import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, CLIPVisionModel
from mobile_sam import sam_model_registry

# ================= 1. Konfiguracja =================
# Teraz lista: zrób dla wszystkich na raz
MODELS_TO_RUN = ['dino_small', 'dino_base', 'clip']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Upewnij się, że masz te foldery (wynik skryptu split_data)
DATA_SPLITS = {
    'train': './fewshot_train/',
    'test': './fewshot_test/'
}

# ================= 2. Główna pętla =================
for model_name in MODELS_TO_RUN:
    print(f"\n{'=' * 60}")
    print(f"ROZPOCZYNAM PRECOMPUTING DLA MODELU: {model_name.upper()}")
    print(f"{'=' * 60}")

    # Konfiguracja parametrów dla danego modelu
    if model_name == 'dino_small':
        model_path = 'facebook/dinov2-small'
        vision_model = AutoModel.from_pretrained(model_path).to(DEVICE)
        FEATURE_KEY = 'dino_feature'
    elif model_name == 'dino_base':
        model_path = 'facebook/dinov2-base'
        vision_model = AutoModel.from_pretrained(model_path).to(DEVICE)
        FEATURE_KEY = 'dino_feature'
    elif model_name == 'clip':
        vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16").to(DEVICE)
        FEATURE_KEY = 'clip_feature'

    vision_model.eval()

    # MobileSAM Encoder (wspólny)
    mobile_sam = sam_model_registry["vit_t"](checkpoint="mobile_sam.pt").to(DEVICE)
    mobile_sam.eval()

    # Przetwarzanie zbiorów (Train i Test)
    for split_name, data_dir in DATA_SPLITS.items():
        output_dir = f'./precomputed_{split_name}_{model_name}/'
        os.makedirs(output_dir, exist_ok=True)

        classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

        with torch.no_grad():
            for class_name in tqdm(classes, desc=f"Klasy ({model_name} | {split_name})"):
                class_path = os.path.join(data_dir, class_name)
                images = [f for f in os.listdir(class_path) if f.endswith('.jpg')]
                if len(images) < 2: continue

                os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

                for img_name in images:
                    # Obraz
                    img_p = os.path.join(class_path, img_name)
                    image = cv2.imread(img_p)
                    if image is None: continue
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Maska
                    mask_p = os.path.join(class_path, img_name.replace('.jpg', '.png'))
                    mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
                    if mask is None: continue
                    mask_tensor = torch.tensor(cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST),
                                               dtype=torch.float32) / 255.0

                    # Cechy wizualne
                    img_tensor_vis = torch.from_numpy(cv2.resize(image_rgb, (224, 224))).permute(2, 0, 1).unsqueeze(
                        0).float() / 255.0
                    img_tensor_vis = img_tensor_vis.to(DEVICE)

                    if 'dino' in model_name:
                        features = vision_model(img_tensor_vis).last_hidden_state[:, 0, :]
                    else:
                        features = vision_model(img_tensor_vis).pooler_output

                    # Cechy SAM
                    img_tensor_sam = torch.from_numpy(cv2.resize(image_rgb, (1024, 1024))).permute(2, 0, 1).unsqueeze(
                        0).float() / 255.0
                    img_tensor_sam = img_tensor_sam.to(DEVICE)
                    sam_features = mobile_sam.image_encoder(img_tensor_sam)

                    # Zapis
                    torch.save({
                        FEATURE_KEY: features.cpu(),
                        'sam_feature': sam_features.cpu(),
                        'gt_mask': mask_tensor.cpu()
                    }, os.path.join(output_dir, class_name, img_name.replace('.jpg', '.pt')))

print("\nWSZYSTKIE MODELE OBLICZONE PRAWIDŁOWO!")