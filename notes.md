# Model Logs
## Model A1
### 🏗️ Model Architecture
| Layer Type        | Output Shape          | Kernel/Stride | Activation | Notes |
|------------------|----------------------|--------------|------------|-------|
| Conv2D (1 → 32) | `(32, 256, 256)`      | `3x3 / 1`    | ReLU       | Extracts low-level features |
| BatchNorm2D(32)  | `(32, 256, 256)`      | -            | -          | Normalizes activations |
| MaxPool2D       | `(32, 128, 128)`      | `2x2 / 2`    | -          | Downsamples |
| Conv2D (32 → 64)| `(64, 128, 128)`      | `3x3 / 1`    | ReLU       | Extracts deeper features |
| BatchNorm2D(64)  | `(64, 128, 128)`      | -            | -          | Improves training stability |
| MaxPool2D       | `(64, 64, 64)`        | `2x2 / 2`    | -          | Reduces spatial size |
| Conv2D (64 → 128)| `(128, 64, 64)`      | `3x3 / 1`    | ReLU       | Higher-level features |
| BatchNorm2D(128)| `(128, 64, 64)`       | -            | -          | Prevents internal covariate shift |
| MaxPool2D       | `(128, 32, 32)`       | `2x2 / 2`    | -          | |
| Fully Connected | `(256)`               | -            | ReLU       | Dense features |
| Dropout (0.5)   | `(256)`               | -            | -          | Prevents overfitting |
| Fully Connected | `(8)`                 | -            | Softmax    | Outputs class probabilities |

### ⚙️ Hyperparameters
| Parameter    | Value   |
|-------------|--------|
| Batch Size  | 32     |
| Learning Rate | 0.001  |
| Epochs      | 10     |
| Weight Decay | 1e-5   |

### 📊 Results
| Epoch | Loss   | Accuracy |
|-------|--------|----------|
| 1     | 1.0572 | 0.6546   |
| 2     |        |          |
| 3     |        |          |
| 4     |        |          |
| 5     |        |          |
| 6     |        |          |
| 7     |        |          |
| 8     |        |          |
| 9     |        |          |
| 10    |        |          |

### 📝 Observations & Adjustments