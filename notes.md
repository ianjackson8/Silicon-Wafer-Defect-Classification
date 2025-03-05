# Model Logs
Definitions:
- Model A: Convolutional Neural Network 

Table of Contents
- [Model A1](#model-a1)
- [Model A2](#model-a2)

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

- Convolutional Layers: 4
- Fully connected layers: 2
- Activation: ReLU
- Dropout: 0.5
- Loss Function: Cross Entropy Loss
- Optimizer: Adam (lr=0.001)

### ⚙️ Hyperparameters
| Parameter    | Value   |
|-------------|--------|
| Batch Size  | 32     |
| Learning Rate | 0.001  |
| Epochs      | 10     |
| Weight Decay | 1e-5   |

### 📊 Results
| Experiment | Epochs | Train Loss | Train Accuracy | Test Accuracy | F1-Score | Avg. Inference Time | Notes |
|------------|--------|------------|----------------|---------------|----------|---------------------|-------|
| Exp-1      | 10     | 0.3042     | 89.07%         | 65.3154%      | 0.6080   | 0.6699 s            | Initial test |
| Exp-2      | 20     | 0.1184     | 96.36%         | 49.5060%      | 0.4086   | 0.6591 s            | Increased epochs |
| Exp-3      |        |            |                |               |          |                     |       |

Training Runtime: 
- Exp1: 4:08:35
- Exp2: N/A

### 📝 Observations & Adjustments
- **Exp-1** Initial test: accuracy, F1, and inference time too low -> double epochs
- **Exp-2** Train accuracy incr, test accuracy dec therefore indicative of overfitting 
- change padding to be center instead of top-left

## Model A2