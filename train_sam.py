import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
from mobile_sam import sam_model_registry

# ================= 1. Konfiguracja =================
# Teraz lista: trenujemy wszystko po kolei
MODELS_TO_TRAIN = ['dino_small', 'dino_base', 'dino_large', 'clip']
EPOCHS = 30000
LR = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs('./models', exist_ok=True)


# ================= 2. Architektura =================
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

    # ================= 3. Główna pętla treningowa =================


for model_name in MODELS_TO_TRAIN:
    print(f"\n{'=' * 60}")
    print(f"ROZPOCZYNAM TRENING DLA: {model_name.upper()}")
    print(f"{'=' * 60}")

    # Ustawienia dla danego wariantu
    if model_name == 'dino_small':
        FEATURES_DIR = './precomputed_train_dino_small/'
        INPUT_DIM, FEATURE_KEY = 384, 'dino_feature'
    elif model_name == 'dino_base':
        FEATURES_DIR = './precomputed_train_dino_base/'
        INPUT_DIM, FEATURE_KEY = 768, 'dino_feature'
    elif model_name == 'dino_large':
        FEATURES_DIR = './precomputed_train_dino_large/'
        INPUT_DIM, FEATURE_KEY = 1024, 'dino_feature'
    elif model_name == 'clip':
        FEATURES_DIR = './precomputed_train_clip/'
        INPUT_DIM, FEATURE_KEY = 768, 'clip_feature'

    adapter = VisualPromptAdapter(input_dim=INPUT_DIM).to(DEVICE)
    optimizer = optim.Adam(adapter.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    # Ładowanie MobileSAM
    mobile_sam = sam_model_registry["vit_t"](checkpoint="mobile_sam.pt").to(DEVICE)
    mobile_sam.eval()
    mask_decoder = mobile_sam.mask_decoder


    def get_random_pair():
        classes = os.listdir(FEATURES_DIR)
        chosen_class = random.choice(classes)
        files = os.listdir(os.path.join(FEATURES_DIR, chosen_class))
        if len(files) < 2: return None, None
        ref_file, query_file = random.sample(files, 2)
        ref_data = torch.load(os.path.join(FEATURES_DIR, chosen_class, ref_file))
        query_data = torch.load(os.path.join(FEATURES_DIR, chosen_class, query_file))
        return ref_data, query_data


    adapter.train()
    for epoch in range(EPOCHS + 1):
        ref_data, query_data = get_random_pair()
        if ref_data is None: continue

        support_feature = ref_data[FEATURE_KEY].to(DEVICE)
        query_sam_features = query_data['sam_feature'].to(DEVICE)
        gt_mask = query_data['gt_mask'].unsqueeze(0).unsqueeze(0).to(DEVICE)

        optimizer.zero_grad()
        sam_prompt = adapter(support_feature)

        # Dense embeddings są wspólne
        dense_embeddings = mobile_sam.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            1, -1, mobile_sam.image_encoder.img_size // 16, mobile_sam.image_encoder.img_size // 16)

        pred_masks, _ = mask_decoder(
            image_embeddings=query_sam_features,
            image_pe=mobile_sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sam_prompt,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )

        loss = loss_fn(pred_masks, gt_mask)
        loss.backward()
        optimizer.step()

        if epoch % 5000 == 0:
            print(f"[{model_name.upper()}] Krok: {epoch:05d}/{EPOCHS} | Loss: {loss.item():.4f}")

        if epoch > 0 and (epoch % 15000 == 0 or epoch == EPOCHS):
            save_name = f"./models/adapter_{model_name}_epoka_{epoch}.pth"
            torch.save(adapter.state_dict(), save_name)
            print(f"----> Zapisano: {save_name}")

print("\nWSZYSTKIE MODELE WYTRENOWANE PRAWIDŁOWO!")
