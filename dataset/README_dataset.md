# Dataset

The dataset is not included in this repository due to its size. Here is how to reconstruct it.

## Download

👉 https://www.kaggle.com/datasets/mostafaabla/garbage-classification

```bash
pip install kaggle
kaggle datasets download -d mostafaabla/garbage-classification
unzip garbage-classification.zip
```

## Classes removed

These classes were removed as they are not relevant in a university context:

| Class | Reason |
|---|---|
| `clothes` | Out of scope |
| `shoes` | Out of scope |
| `battery` | Dedicated collection stream |
| `brown-glass` | Dedicated collection stream |
| `white-glass` | Dedicated collection stream |
| `green-glass` | Dedicated collection stream |

## Classes kept

| Class | Label |
|---|---|
| `plastic` | recyclable |
| `paper` | recyclable |
| `cardboard` | recyclable |
| `metal` | recyclable |
| `biological` | non_recyclable |
| `trash` | non_recyclable |

## Expected folder structure

```
dataset/
  recyclable/
    img1.jpg
    ...
  non_recyclable/
    img1.jpg
    ...
```

## Final statistics

| Split | Images | Recyclable | Non-recyclable |
|---|---|---|---|
| Train (80%) | 7,065 | ~4,028 | ~3,037 |
| Val (10%) | 883 | ~503 | ~380 |
| Test (10%) | 884 | ~504 | ~380 |
| **Total** | **8,832** | **4,027** | **3,038** |
