# Manufacturing Surface Defect Detection

Deep learning classifier for steel surface defects, trained on the **NEU Surface Defect Database (NEU-DET)** ([Kaggle](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database/data)). Built for ISEM 505.

An EfficientNetB0 backbone (ImageNet weights) is trained in two stages — frozen feature extraction, then fine-tuning from layer 200 — and the stage with the better validation accuracy is promoted to the final model.

## Results

Final model: **Stage 2 (fine-tuned)**, selected on validation accuracy with macro F1 as tie-breaker.

| Split | Accuracy | Macro F1 |
|---|---|---|
| Validation | 98.96% | 0.9896 |
| Test | 93.06% | 0.9301 |

Test set: 335/360 images correct across 6 defect classes.

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Crazing | 0.952 | 1.000 | 0.976 |
| Inclusion | 0.934 | 0.950 | 0.942 |
| Patches | 1.000 | 1.000 | 1.000 |
| Pitted Surface | 1.000 | 0.833 | 0.909 |
| Rolled In Scale | 0.800 | 1.000 | 0.889 |
| Scratches | 0.941 | 0.800 | 0.865 |

The weakest pair is Scratches vs. Rolled In Scale — see `artifacts/plots/07_test_confusion_matrix.png`.

## Layout

```
notebooks/steel_surface_defect_deep_learning.ipynb   # full pipeline: EDA -> training -> evaluation
artifacts/model/steel_defect_classifier.keras        # final trained model
artifacts/metrics/                                   # test metrics, classification report, confusion matrix, training history
artifacts/metadata/                                  # model metadata + split manifest
artifacts/predictions/                               # per-image test predictions
artifacts/plots/                                     # 9 figures (splits, class examples, curves, confusion matrix)
Steel_Defect_Classification_ISEM505.pptx             # project presentation
```

## Setup

The dataset and intermediate checkpoints are gitignored. To reproduce:

1. Download the dataset from [Kaggle: NEU Surface Defect Database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database/data) and unzip it so images land under `data/NEU-DET/`.
2. Run the notebook — it builds the stratified `data/train`, `data/validation`, and `data/test` splits (1152 / 288 / 360 images, `random_state=42`).

```bash
pip install tensorflow==2.20.0 scikit-learn pandas matplotlib seaborn pillow
jupyter lab notebooks/steel_surface_defect_deep_learning.ipynb
```

## Inference

Normalisation is a layer inside the model, so raw images need no external preprocessing — just resize to 224x224 and convert grayscale to RGB.

```python
import tensorflow as tf, numpy as np

model = tf.keras.models.load_model("artifacts/model/steel_defect_classifier.keras")
classes = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

img = tf.keras.utils.load_img("path/to/image.jpg", target_size=(224, 224), color_mode="rgb")
probs = model.predict(np.expand_dims(tf.keras.utils.img_to_array(img), 0))[0]
print(classes[int(probs.argmax())], float(probs.max()))
```

## Details

- **Framework:** TensorFlow 2.20.0 / Keras 3.13.2
- **Input:** 224x224x3, batch size 32
- **Augmentation:** training split only — flip, rotation, zoom, translation, contrast
- **Epochs trained:** 29
- **Dataset:** [NEU Surface Defect Database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database/data) (Kaggle)
