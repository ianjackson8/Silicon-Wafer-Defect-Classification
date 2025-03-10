# Model Logs
Definitions:
- Model A: Convolutional Neural Network 

Table of Contents
- [Model A1](#model-a1)
- [Model A2](#model-a2)

## Model A1
### 🏗️ Model Architecture
| Layer Type        | Output Shape          | Kernel/Stride| Activation | Notes |
|-------------------|-----------------------|--------------|------------|-------|
| Conv2D (1 → 32)   | `(32, 256, 256)`      | `3x3 / 1`    | ReLU       | Extracts low-level features |
| BatchNorm2D(32)   | `(32, 256, 256)`      | -            | -          | Normalizes activations |
| MaxPool2D         | `(32, 128, 128)`      | `2x2 / 2`    | -          | Downsamples |
| Conv2D (32 → 64)  | `(64, 128, 128)`      | `3x3 / 1`    | ReLU       | Extracts deeper features |
| BatchNorm2D(64)   | `(64, 128, 128)`      | -            | -          | Improves training stability |
| MaxPool2D         | `(64, 64, 64)`        | `2x2 / 2`    | -          | Reduces spatial size |
| Conv2D (64 → 128) | `(128, 64, 64)`       | `3x3 / 1`    | ReLU       | Higher-level features |
| BatchNorm2D(128)  | `(128, 64, 64)`       | -            | -          | Prevents internal covariate shift |
| MaxPool2D         | `(128, 32, 32)`       | `2x2 / 2`    | -          | |
| Conv2D (128 → 256)| `(256, 32, 32)`       | `3x3 / 1`    | ReLU       | |
| BatchNorm2D(256)  | `(256, 32, 32)`       | -            | -          | |
| MaxPool2D         | `(256, 16, 16)`       | `2x2 / 2`    | -          | |
| Fully Connected   | `(256)`               | -            | ReLU       | Dense features |
| Dropout (0.5)     | `(256)`               | -            | -          | Prevents overfitting |
| Fully Connected   | `(8)`                 | -            | Softmax    | Outputs class probabilities |

- Convolutional Layers: 4
- Fully connected layers: 2
- Activation: ReLU
- Dropout: 0.6
- Loss Function: Cross Entropy Loss
- Optimizer: Adam (lr=0.001)

### ⚙️ Hyperparameters
| Parameter    | Value   |
|-------------|--------|
| Batch Size  | 32     |
| Learning Rate | 0.001  |
| Epochs      | 30     |
| Weight Decay | 1e-5   |

### 📊 Results
| Experiment | Epochs | Train Loss | Train Accuracy | Test Accuracy | F1-Score | Avg. Inference Time | Notes |
|------------|--------|------------|----------------|---------------|----------|---------------------|-------|
| Exp-1      | 10     | 0.3042     | 89.07%         | 65.3154%      | 0.6080   | 0.0053 s            | Initial test |
| Exp-2      | 20     | 0.1184     | 96.36%         | 49.5060%      | 0.4086   | 0.0012 s            | Increased epochs |
| Exp-3      | 10     | 0.3160     | 88.17%         | 63.0352%      | 0.5871   | 0.0011 s            | Dropout 0.5→0.6 |
| Exp-4      | 10     | 0.2979     | 89.23%         | 61.4011%      | 0.5900   | 0.0012 s            | L2 regularization |
| Exp-5      | 10     | 0.2425     | 90.89%         | 66.0122%      | 0.6312   | 0.0018 s            | StepLR ($\gamma$=0.5/5 epoch) |
| Exp-6      | 10     | 0.2868     | 89.54%         | 60.9577%      | 0.5579   | 0.0012 s            | Dropout 0.5→0.6, L2 regularization, StepLR ($\gamma$=0.5/5 epoch) |
| Exp-7      | 20     | 0.1616     | 94.36%         | 68.7611%      | 0.6735   | 0.0016 s            | Increased epochs |
| Exp-8      | 30     | 0.0532     | 98.21%         | 69.2931%      | 0.7028   | 0.0089 s            | Increased epochs |
| Exp-9      | 40     | 0.0494     | 98.33%         | 68.8118%      | 0.6986   | 0.0050 s            | Increased epochs |

### 📝 Observations & Adjustments
- **Exp-1** Initial test: accuracy, F1, and inference time too low → double epochs
- **Exp-2** Train accuracy incr, test accuracy dec therefore indicative of overfitting 
- **Exp-3 → Exp-5** Test anti-overfitting methods
- **Exp-6** Try all overfitting methods in one
- **Exp-7 → Exp-9** Increase epochs until decrease in accuracy

### 🛠️ Tested overfitting methods
- [x] Increase dropout rate ✅
- [x] Add weight decay (L2 regularization) 1e-4 ✅
- [ ] Use learning rate scheduling 
  - [x] StepLR ✅
  - [ ] ReduceLROnPlateau
  - [ ] CosineAnnealing
- [ ] Data augmentation (`torchvision.transform`)
- [x] Reduce number of filters or layers (A1→A2)

## Model A2
### 🏗️ Model Architecture
| Layer Type        | Output Shape          | Kernel/Stride| Activation | Notes |
|-------------------|-----------------------|--------------|------------|-------|
| Conv2D (1 → 32)   | `(32, 256, 256)`      | `3x3 / 1`    | ReLU       |  |
| BatchNorm2D(32)   | `(32, 256, 256)`      | -            | -          |  |
| MaxPool2D         | `(32, 85,  85)`       | `3x3 / 3`    | -          | Early aggressive downsampling |
| Conv2D (32 → 64)  | `(64, 85,  85)`       | `3x3 / 1`    | ReLU       |  |
| BatchNorm2D(64)   | `(64, 85,  85)`       | -            | -          |  |
| MaxPool2D         | `(64, 42, 42)`        | `2x2 / 2`    | -          |  |
| Conv2D (64 → 128) | `(128, 42, 42)`       | `3x3 / 1`    | ReLU       |  |
| BatchNorm2D(128)  | `(128, 42, 42)`       | -            | -          |  |
| MaxPool2D         | `(128, 21, 21)`       | `2x2 / 2`    | -          |  |
| Fully Connected   | `(256)`               | -            | ReLU       |  |
| Dropout (0.6)     | `(256)`               | -            | -          |  |
| Fully Connected   | `(8)`                 | -            | Softmax    |  |

- Convolutional Layers: 3
- Fully connected layers: 2
- Activation: ReLU
- Dropout: 0.6
- Loss Function: Cross Entropy Loss
- Optimizer: Adam (lr=0.001)

### ⚙️ Hyperparameters
| Parameter    | Value   |
|-------------|--------|
| Batch Size  | 32     |
| Learning Rate | 0.001  |
| Epochs      | 30     |
| Weight Decay | 1e-5   |

### 📊 Results
| Experiment | Epochs | Train Loss | Train Accuracy | Test Accuracy | F1-Score | Avg. Inference Time | Notes |
|------------|--------|------------|----------------|---------------|----------|---------------------|-------|
| Exp-1      | 30     | 0.0000     | 00.00%         | 00.0000%      | 0.0000   | 0.0000 s            |       |
| Exp-2      | 00     | 0.0000     | 00.00%         | 00.0000%      | 0.0000   | 0.0000 s            |       |
| Exp-3      | 00     | 0.0000     | 00.00%         | 00.0000%      | 0.0000   | 0.0000 s            |       |
| Exp-4      | 00     | 0.0000     | 00.00%         | 00.0000%      | 0.0000   | 0.0000 s            |       |
| Exp-5      | 00     | 0.0000     | 00.00%         | 00.0000%      | 0.0000   | 0.0000 s            |       |

### 📝 Observations & Adjustments
- c
- change padding to be center instead of top-left

### 🛠️ Tested overfitting methods
- [ ] Increase dropout rate 
- [ ] Use learning rate scheduling 
  - [ ] ReduceLROnPlateau
  - [ ] CosineAnnealing
- [ ] Data augmentation (`torchvision.transform`)
- [ ] Reduce number of filters or layers (A2→A3)