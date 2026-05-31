import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
MODEL_PATH  = "./results/model.pth"
IMAGE_SIZE  = 224
CLASSES     = ["non_recyclable", "recyclable"]
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
#  TRANSFORM
# ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ─────────────────────────────────────────────
#  CHARGEMENT DU MODÈLE
# ─────────────────────────────────────────────
def load_model():
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, len(CLASSES))
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# ─────────────────────────────────────────────
#  INFÉRENCE
# ─────────────────────────────────────────────
def predict(image_path: str, model) -> None:
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Erreur lors du chargement de l'image : {e}")
        return

    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1)[0]
        pred   = torch.argmax(probs).item()

    label      = CLASSES[pred]
    confidence = probs[pred].item() * 100

    emoji = "♻️ " if label == "recyclable" else "🚯"
    print(f"\n{emoji}  {label.upper()}")
    print(f"   Confiance : {confidence:.1f}%")
    print(f"   Recyclable     : {probs[1].item()*100:.1f}%")
    print(f"   Non recyclable : {probs[0].item()*100:.1f}%\n")

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PlasticSense — Classification d'un déchet"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Chemin vers l'image à classifier"
    )
    args = parser.parse_args()

    print(f"Chargement du modèle depuis {MODEL_PATH}...")
    model = load_model()
    print(f"Analyse de : {args.image}")
    predict(args.image, model)
