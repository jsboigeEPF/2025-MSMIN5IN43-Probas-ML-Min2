import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import string
import time

# Import de l'architecture partagée
# On essaye l'import local (si lancé depuis backend/) ou relatif
try:
    from architecture import CRNN
except ImportError:
    from backend.architecture import CRNN

# Configuration
# On se base sur le fait qu'on lance le script depuis le dossier 3_4_captcha/ ou backend/
if os.path.exists("data"):
    DATA_ROOT = "data"
elif os.path.exists("../data"):
    DATA_ROOT = "../data"
else:
    raise FileNotFoundError("Dossier data introuvable")

IMAGES_DIR = os.path.join(DATA_ROOT, "images")
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
VAL_CSV = os.path.join(DATA_ROOT, "val.csv")
"""
MODEL_SAVE_PATH = (
    "model.pth" if os.path.exists("model.pth") else "backend/model.pth"
)
if (
    not os.path.exists(os.path.dirname(MODEL_SAVE_PATH))
    and os.path.dirname(MODEL_SAVE_PATH) != ""
):
    # Fallback si on est dans 3_4_captcha/
    MODEL_SAVE_PATH = "backend/model.pth"
"""
MODEL_SAVE_PATH = "model.pth"

IMG_WIDTH = 200
IMG_HEIGHT = 64
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 15
ALPHABET = string.ascii_uppercase

# Mapping caractères <-> index
# CTC Blank est souvent 0. Donc A=1, B=2, ...
CHAR2IDX = {char: idx + 1 for idx, char in enumerate(ALPHABET)}
IDX2CHAR = {idx + 1: char for idx, char in enumerate(ALPHABET)}


class CaptchaDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = row["filename"]
        label_str = row["Label"]

        img_path = os.path.join(self.root_dir, img_name)
        try:
            # Ouverture en niveau de gris (L)
            image = Image.open(img_path).convert("L")
        except Exception as e:
            print(f"Erreur chargement image {img_path}: {e}")
            # Image noire en fallback
            image = Image.new("L", (IMG_WIDTH, IMG_HEIGHT))

        if self.transform:
            image = self.transform(image)

        # Encodage du label
        label = torch.tensor(
            [CHAR2IDX[c] for c in label_str if c in CHAR2IDX], dtype=torch.long
        )
        return image, label


def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, 0)  # [Batch, 1, H, W]

    # Pour CTC Loss, on a besoin des targets concaténées et de leurs longueurs
    target_lengths = torch.tensor([len(t) for t in labels], dtype=torch.long)
    targets = torch.cat(labels)

    return images, targets, target_lengths


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for batch_idx, (images, targets, target_lengths) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        optimizer.zero_grad()

        # Forward
        # Output du modèle: [Batch, TimeSteps, NumClasses] (LogSoftmaxed)
        preds = model(images)

        # CTC Loss attend [TimeSteps, Batch, NumClasses]
        preds_permuted = preds.permute(1, 0, 2)

        # Input lengths : tous egaux à la largeur temporelle de sortie du CNN/RNN
        # Ici c'est 50 (cf architecture.py: W // 4 = 200 // 4 = 50)
        # Mais on récupère dynamiquement
        input_lengths = torch.full(
            size=(images.size(0),), fill_value=preds.size(1), dtype=torch.long
        ).to(device)

        loss = criterion(preds_permuted, targets, input_lengths, target_lengths)

        loss.backward()
        # Clip gradients pour éviter l'explosion (typique RNN)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, targets, target_lengths in loader:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            preds = model(images)
            preds_permuted = preds.permute(1, 0, 2)
            input_lengths = torch.full(
                size=(images.size(0),),
                fill_value=preds.size(1),
                dtype=torch.long,
            ).to(device)

            loss = criterion(
                preds_permuted, targets, input_lengths, target_lengths
            )
            total_loss += loss.item()
    return total_loss / len(loader)


def decode_prediction(preds):
    # preds: [TimeSteps, NumClasses] or [Batch, TimeSteps, NumClasses]
    # Simple Greedy Decoder
    if preds.dim() == 3:
        preds = preds.argmax(dim=2)  # [Batch, TimeSteps]

    decoded_batch = []
    for p in preds:
        p = p.cpu().numpy()
        decoded_str = ""
        for i in range(len(p)):
            if p[i] != 0 and (i == 0 or p[i] != p[i - 1]):
                decoded_str += IDX2CHAR.get(p[i], "")
        decoded_batch.append(decoded_str)
    return decoded_batch


def main():
    print("Initialisation de l'entraînement...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Transforms
    transform = transforms.Compose(
        [
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # [-1, 1]
        ]
    )

    # Datasets
    train_dataset = CaptchaDataset(TRAIN_CSV, IMAGES_DIR, transform=transform)
    val_dataset = CaptchaDataset(VAL_CSV, IMAGES_DIR, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    # Model
    model = CRNN(num_chars=len(ALPHABET), hidden_size=256)
    model.to(device)

    # Loss & Optimizer
    # blank=0 car on a mappé les charactères à partir de 1
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=3, factor=0.5
    )

    best_val_loss = float("inf")

    print(f"Début de l'entraînement pour {EPOCHS} époques.")

    for epoch in range(EPOCHS):
        start_time = time.time()
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss = val_epoch(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        duration = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {duration:.1f}s"
        )

        # Test visuel rapide sur le premier batch de validation
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                imgs, _, _ = next(iter(val_loader))
                imgs = imgs.to(device)
                preds = model(imgs)
                decoded = decode_prediction(preds)
                print(f"  Exemple Pred: {decoded[0]}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Sauvegarde du state dict uniquement
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> Modèle sauvegardé dans {MODEL_SAVE_PATH}")

    print("Entraînement terminé.")


if __name__ == "__main__":
    main()
