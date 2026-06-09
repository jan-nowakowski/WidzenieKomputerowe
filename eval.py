import os
import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mobile_sam import sam_model_registry

# ================= 1. Konfiguracja =================
EPOCH_TO_LOAD = 30000  # Numer epoki, którą chcesz przetestować
DATA_DIR = './fewshot_test/'  # Folder z obrazkami testowymi
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Lista modeli do ewaluacji
MODELS_TO_TEST = ['dino_small', 'dino_base', 'clip']


# ================= 2. Architektura Adaptera =================
class VisualPromptAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x).unsqueeze(1)


def calculate_iou(pred_mask, gt_mask):
    pred = (pred_mask > 0.0).float()
    gt = gt_mask.float()
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection
    if union == 0: return 1.0
    return (intersection / union).item()


# ================= 3. Inicjalizacja wspólnego modelu SAM =================
print("Ładowanie dekodera MobileSAM...")
mobile_sam = sam_model_registry["vit_t"](checkpoint="mobile_sam.pt").to(DEVICE)
mobile_sam.eval()
mask_decoder = mobile_sam.mask_decoder

final_results = {}

# ================= 4. Główna Pętla Ewaluacyjna =================
for MODEL_NAME in MODELS_TO_TEST:
    print(f"\n{'=' * 50}\nEWALUACJA: {MODEL_NAME.upper()}\n{'=' * 50}")

    # Mapowanie ścieżek
    if MODEL_NAME == 'dino_small':
        FEATURES_DIR = './precomputed_test_dino_small/'
        INPUT_DIM, FEATURE_KEY = 384, 'dino_feature'
    elif MODEL_NAME == 'dino_base':
        FEATURES_DIR = './precomputed_test_dino_base/'
        INPUT_DIM, FEATURE_KEY = 768, 'dino_feature'
    elif MODEL_NAME == 'clip':
        FEATURES_DIR = './precomputed_test_clip/'
        INPUT_DIM, FEATURE_KEY = 768, 'clip_feature'

    WEIGHTS_PATH = f'./models/adapter_{MODEL_NAME}_epoka_{EPOCH_TO_LOAD}.pth'
    OUTPUT_VIS_DIR = f'./results_vis_{MODEL_NAME}/'
    os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)

    if not os.path.exists(WEIGHTS_PATH):
        print(f"❌ Brak wag: {WEIGHTS_PATH}. Pomiń.")
        continue

    # Ładowanie adaptera
    adapter = VisualPromptAdapter(input_dim=INPUT_DIM).to(DEVICE)
    adapter.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    adapter.eval()

    classes = [d for d in os.listdir(FEATURES_DIR) if os.path.isdir(os.path.join(FEATURES_DIR, d))]
    total_iou = 0.0
    count = 0

    with torch.no_grad():
        for class_name in classes:
            feat_class_dir = os.path.join(FEATURES_DIR, class_name)
            img_class_dir = os.path.join(DATA_DIR, class_name)
            files = [f for f in os.listdir(feat_class_dir) if f.endswith('.pt')]
            if len(files) < 2: continue

            ref_file = files[0]
            ref_data = torch.load(os.path.join(feat_class_dir, ref_file))
            support_feature = ref_data[FEATURE_KEY].to(DEVICE)

            sam_prompt = adapter(support_feature)
            dense_embeddings = mobile_sam.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                1, -1, mobile_sam.image_encoder.img_size // 16, mobile_sam.image_encoder.img_size // 16)

            for query_file in files[1:]:
                query_data = torch.load(os.path.join(feat_class_dir, query_file))
                query_sam_features = query_data['sam_feature'].to(DEVICE)
                gt_mask_tensor = query_data['gt_mask'].to(DEVICE)

                pred_masks, _ = mask_decoder(
                    image_embeddings=query_sam_features,
                    image_pe=mobile_sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sam_prompt,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False
                )

                iou = calculate_iou(pred_masks[0, 0], gt_mask_tensor)
                total_iou += iou
                count += 1

                # Zapisujemy tylko pierwsze 2 predykcje z każdej klasy dla wizualizacji
                if count % 20 == 0:
                    ref_img = cv2.cvtColor(cv2.imread(os.path.join(img_class_dir, ref_file.replace('.pt', '.jpg'))),
                                           cv2.COLOR_BGR2RGB)
                    query_img = cv2.cvtColor(cv2.imread(os.path.join(img_class_dir, query_file.replace('.pt', '.jpg'))),
                                             cv2.COLOR_BGR2RGB)

                    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
                    axs[0].imshow(ref_img);
                    axs[0].set_title("Referencja")
                    axs[1].imshow(query_img);
                    axs[1].set_title("Cel")
                    axs[2].imshow(gt_mask_tensor.cpu(), cmap='gray');
                    axs[2].set_title("Ground Truth")
                    axs[3].imshow((pred_masks[0, 0].cpu() > 0.0).numpy(), cmap='jet');
                    axs[3].set_title(f"IoU: {iou:.2f}")
                    for ax in axs: ax.axis('off')
                    plt.savefig(os.path.join(OUTPUT_VIS_DIR, f"{class_name}_{query_file.replace('.pt', '.png')}"))
                    plt.close(fig)

    final_results[MODEL_NAME] = total_iou / count if count > 0 else 0
    print(f" Wynik: {final_results[MODEL_NAME]:.4f}")

# ================= 5. Podsumowanie =================
print("\n" + "*" * 50)
print("              WYNIKI KOŃCOWE (mIoU)")
print("*" * 50)
for model, score in final_results.items():
    print(f"{model.upper():<20} |  {score:.4f}")
print("*" * 50)