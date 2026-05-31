# 🗑️ PlasticSense — Smart Waste Sorting Bin

![Accuracy](https://img.shields.io/badge/Accuracy-97.7%25-brightgreen)
![F1 Score](https://img.shields.io/badge/F1%20Score-0.977-brightgreen)
![Model](https://img.shields.io/badge/Model-MobileNetV3--Small-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%204-red)

> Prototype de poubelle intelligente développé à **Sorbonne Université**.  
> Une caméra analyse le déchet posé sur un plateau circulaire, un modèle IA classifie le déchet en **recyclable** ou **non recyclable**, et un servomoteur oriente automatiquement le plateau vers le bon bac.

---

## 🎯 Résultats

| Métrique | Valeur |
|---|---|
| Meilleure Val Accuracy | 98.4% |
| **Test Accuracy** | **97.7%** |
| F1 Score | 0.977 |
| Objectif initial | 90% |

### Progression historique

| Version | Accuracy | Description |
|---|---|---|
| V1 | 68.4% | Modèle initial |
| V2 | 73.6% | Premier nettoyage dataset |
| V3 | 84.6% | Ajout photos iPhone réelles |
| **V4** | **97.7%** | Pipeline complet optimisé |

---

## 🏗️ Architecture

```
Déchet posé sur le plateau
        ↓
    Caméra (capture)
        ↓
MobileNetV3-Small (classification)
        ↓
Servomoteur (bascule le plateau)
        ↓
Recyclable ♻️  /  Non recyclable 🚯
```

---

## 📁 Structure du repo

```
PlasticSense/
  train.py          # Entraînement du modèle
  inference.py      # Inférence sur une image
  requirements.txt  # Dépendances
  results/
    model.pth       # Modèle entraîné (97.7%)
  dataset/
    README.md       # Instructions pour télécharger le dataset
```

---

## ⚙️ Installation

```bash
git clone https://github.com/adnanegrb/PlasticSense.git
cd PlasticSense
pip install -r requirements.txt
```

---

## 🚀 Utilisation

### Entraîner le modèle

```bash
python train.py
```

Le modèle sera sauvegardé dans `results/model.pth`.

### Tester sur une image

```bash
python inference.py --image chemin/vers/image.jpg
```

Exemple :
```bash
python inference.py --image test.jpg
# → ♻️  RECYCLABLE (confiance : 98.3%)
```

---

## 📊 Dataset

Le dataset utilisé provient de Kaggle — voir `dataset/README.md` pour les instructions de téléchargement.

- **Total** : 8 832 images
- **Recyclable** : 4 027 images (plastique, papier, carton, métal)
- **Non recyclable** : 3 038 images (déchets biologiques, divers)
- **Split** : 80% train / 10% val / 10% test

---

## 🧠 Modèle

- **Architecture** : MobileNetV3-Small (pré-entraîné ImageNet)
- **Fine-tuning** : 3 derniers blocs dégelés
- **Optimizer** : AdamW (weight decay 1e-4)
- **Scheduler** : CosineAnnealingLR
- **Early Stopping** : patience = 3

---

## 🔧 Matériel

- Raspberry Pi 4
- Caméra module CSI
- Servomoteur
- Alimentation 5V

---

## 👨‍💻 Auteur

**Mohammed Adnane Garab** — Sorbonne Université  
[GitHub](https://github.com/adnanegrb)

---

## 📄 Licence

MIT License
