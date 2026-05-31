import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, f1_score, classification_report
import optuna

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
class Config:
    DATA_PATH    = "./dataset"          # recyclable/ et non_recyclable/
    MODEL_PATH   = "./model.pth"
    IMAGE_SIZE   = 224                  # MobileNetV3 préfère 224
    BATCH_SIZE   = 16
    NUM_EPOCHS   = 20
    LR           = 3e-4
    WEIGHT_DECAY = 1e-4
    PATIENCE     = 5                    # early stopping
    DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = Config()
print(f"Device utilisé : {cfg.DEVICE}")

# ─────────────────────────────────────────────
#  TRANSFORMS (augmentation forte pour généraliser)
# ─────────────────────────────────────────────
def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf

# ─────────────────────────────────────────────
#  DATA LOADING avec WeightedSampler (équilibre les classes)
# ─────────────────────────────────────────────
def load_data():
    train_tf, val_tf = get_transforms()

    full_dataset = datasets.ImageFolder(cfg.DATA_PATH, transform=train_tf)
    n = len(full_dataset)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)
    n_test  = n - n_train - n_val

    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    # Applique les transforms val/test sans augmentation
    val_ds.dataset  = copy.deepcopy(full_dataset)
    val_ds.dataset.transform  = val_tf
    test_ds.dataset = copy.deepcopy(full_dataset)
    test_ds.dataset.transform = val_tf

    # WeightedRandomSampler → équilibre les classes déséquilibrées
    targets = [full_dataset.targets[i] for i in train_ds.indices]
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, sampler=sampler, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False,   num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE, shuffle=False,   num_workers=2)

    print(f"\nClasses : {full_dataset.classes}")
    print(f"Train: {n_train} | Val: {n_val} | Test: {n_test}")
    print(f"Distribution : {dict(zip(full_dataset.classes, class_counts))}\n")

    return train_loader, val_loader, test_loader, full_dataset.classes

# ─────────────────────────────────────────────
#  MODÈLE — MobileNetV3-Small pré-entraîné
# ─────────────────────────────────────────────
def build_model(num_classes, dropout=0.3):
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

    # Dégèle les 3 derniers blocs pour fine-tuning profond
    for param in model.parameters():
        param.requires_grad = False
    for param in model.features[-3:].parameters():
        param.requires_grad = True

    # Remplace la tête de classification
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )
    return model.to(cfg.DEVICE)

# ─────────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    losses, preds, labels = [], [], []

    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
            if is_train:
                optimizer.zero_grad()
            out  = model(x)
            loss = criterion(out, y)
            if is_train:
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
            preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            labels.extend(y.cpu().numpy())

    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="macro", zero_division=0)
    return np.mean(losses), acc, f1

# ─────────────────────────────────────────────
#  ENTRAÎNEMENT PRINCIPAL avec Early Stopping
# ─────────────────────────────────────────────
def train(lr=None, weight_decay=None, dropout=None):
    lr           = lr           or cfg.LR
    weight_decay = weight_decay or cfg.WEIGHT_DECAY
    dropout      = dropout      or 0.3

    train_loader, val_loader, test_loader, classes = load_data()
    model     = build_model(len(classes), dropout=dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.NUM_EPOCHS)

    best_val_acc  = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    print("=" * 55)
    print(f"{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Acc':>8} {'Val F1':>7}")
    print("=" * 55)

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        t_loss, t_acc, _    = run_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_acc, v_f1 = run_epoch(model, val_loader,   criterion)
        scheduler.step()

        print(f"{epoch:>6} {t_loss:>11.4f} {t_acc:>9.1%} {v_acc:>8.1%} {v_f1:>7.3f}")

        if v_acc > best_val_acc:
            best_val_acc   = v_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save(best_model_wts, cfg.MODEL_PATH)
            print(f"  ✅ Nouveau meilleur modèle sauvegardé ({best_val_acc:.1%})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.PATIENCE:
                print(f"\n⛔ Early stopping à l'epoch {epoch}")
                break

    # Évaluation finale sur le test set
    model.load_state_dict(best_model_wts)
    _, test_acc, test_f1 = run_epoch(model, test_loader, criterion)
    print(f"\n🏆 Test Accuracy : {test_acc:.1%} | Test F1 : {test_f1:.3f}")
    print(f"Modèle sauvegardé → {cfg.MODEL_PATH}")

    return best_val_acc

# ─────────────────────────────────────────────
#  OPTUNA — optimisation automatique des hyperparamètres
#  Décommente la section ci-dessous pour l'utiliser
# ─────────────────────────────────────────────
def optuna_objective(trial):
    lr           = trial.suggest_float("lr",           1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    dropout      = trial.suggest_float("dropout",      0.1,  0.5)
    cfg.BATCH_SIZE = trial.suggest_categorical("batch_size", [8, 16, 32])
    return train(lr=lr, weight_decay=weight_decay, dropout=dropout)

def run_optuna(n_trials=20):
    study = optuna.create_study(direction="maximize")
    study.optimize(optuna_objective, n_trials=n_trials)
    print("\n🔬 Meilleurs hyperparamètres trouvés :")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"  Meilleure val accuracy: {study.best_value:.1%}")

# ─────────────────────────────────────────────
#  LANCEMENT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # ── Mode normal (recommandé pour commencer) ──
    train()

    # ── Mode Optuna (décommente quand tu veux optimiser) ──
    # run_optuna(n_trials=20)