# Model Logs
Definitions:
- Model A: Convolutional Neural Network 
- Model B: Support Vector Machine
- Model C: K-Nearest Neighbors

Table of Contents
- [Model A1](#model-a1)
- [Model A2](#model-a2)
- [Model A3](#model-a3)
- [Model A4](#model-a4)
- [Model B1](#model-b1)
- [Model B2](#model-b2)
- [Model C1](#model-c1)

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
| Parameter     | Value   |
|---------------|---------|
| Batch Size    | 32      |
| Learning Rate | 0.001   |
| Epochs        | 30      |
| Weight Decay  | 1e-5    |

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
| **Exp-8**  | 30     | 0.0532     | 98.21%         | 69.2931%      | 0.7028   | 0.0089 s            | Increased epochs |
| Exp-9      | 40     | 0.0494     | 98.33%         | 68.8118%      | 0.6986   | 0.0050 s            | Increased epochs |
| Exp-10     | 75     | 0.7372     | 71.35%         | 62.7185%      | 0.5966   | 0.0052 s            | Introduce augmentation (H&V flip, pm 90 deg), LR/100 at 50 epoch, removed scheduler |
| Exp-11     | 75     | 0.4525     | 71.63%         | 65.1001%      | 0.6252   | 0.0052 s            | Use focal loss ($\gamma=2$) |
| Exp-12     | 75     | 0.4464     | 71.76%         | 65.6068%      | 0.6355   | 0.0055 s            | Introduced SWA (lr = 1e-5) |

### 📝 Observations & Adjustments
- **Exp-1** Initial test: accuracy, F1, and inference time too low → double epochs
- **Exp-2** Train accuracy incr, test accuracy dec therefore indicative of overfitting 
- **Exp-3 → Exp-5** Test anti-overfitting methods
- **Exp-6** Try all overfitting methods in one
- **Exp-7 → Exp-9** Increase epochs until decrease in accuracy
- **Exp-10** Introduce augmentation, gets stuck in local min. Removed LR scheduling, 75 epoch. Train & Test accuracy closer

### 🛠️ Tested overfitting methods
- [x] Increase dropout rate ✅
- [x] Add weight decay (L2 regularization) 1e-4 ✅
- [ ] Use learning rate scheduling 
  - [x] StepLR ✅
  - [ ] ReduceLROnPlateau
  - [ ] CosineAnnealing
- [x] Data augmentation (`torchvision.transform`)
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
| Parameter     | Value   |
|---------------|---------|
| Batch Size    | 32      |
| Learning Rate | 0.001   |
| Epochs        | 30      |
| Weight Decay  | 1e-5    |

### 📊 Results
| Experiment | Epochs | Train Loss | Train Accuracy | Test Accuracy | F1-Score | Avg. Inference Time | Notes |
|------------|--------|------------|----------------|---------------|----------|---------------------|-------|
| Exp-1      | 30     | 0.1857     | 92.59%         | 66.9116%      | 0.6382   | 0.0008 s            | Initial run |
| Exp-2      | 30     | 0.2120     | 91.71%         | 65.6575%      | 0.6220   | 0.0007 s            | change padding to center |
| **Exp-3**  | 30     | 0.2234     | 91.85%         | 67.2536%      | 0.6424   | 0.0038 s            | Introduced augmentation (H&V flip) |
| Exp-4      | 40     | 0.2691     | 90.43%         | 66.8862%      | 0.6344   | 0.0008 s            | More epochs |
| Exp-5      | 50     | 0.9358     | 64.52%         | 39.9924%      | 0.3831   | 0.0046 s            | Introduce rotational augmentation |
| Exp-6      | 30     | 0.0000     | 00.00%         | 00.0000%      | 0.0000   | 0.0000 s            |  |
| Exp-7      | 30     | 0.0000     | 00.00%         | 00.0000%      | 0.0000   | 0.0000 s            |  |

### 📝 Observations & Adjustments
- **Exp-1** slightly worse results than A2 counterpart
- **Exp-2** changing padding to center has worse/no effects on results
- **Exp-3** augmentation improved test accuracy, let run for more epochs
- **Exp-4** more epochs no good, add rotational augmentation by pm 90 deg

### 🛠️ Tested overfitting methods
- [x] Increase epochs ❌
- [ ] Increase dropout rate 
- [ ] Use learning rate scheduling 
  - [ ] ReduceLROnPlateau
  - [ ] CosineAnnealing
- [x] Data augmentation (`torchvision.transform`) ✅

## Model A3
### 🏗️ Model Architecture
| Layer Type            | Output Shape         | Kernel/Stride | Activation | Notes                                 |
|-----------------------|----------------------|---------------|------------|---------------------------------------|
| Conv2D (1 → 8)        | `(8, 256, 256)`      | `3x3 / 1`     | ReLU       | Extracts low-level features           |
| BatchNorm2D(8)        | `(8, 256, 256)`      | -             | -          | Normalizes activations                |
| MaxPool2D             | `(8, 128, 128)`      | `2x2 / 2`     | -          | Downsamples spatial dimensions        |
| Conv2D (8 → 16)       | `(16, 128, 128)`     | `3x3 / 1`     | ReLU       | Extracts deeper features              |
| BatchNorm2D(16)       | `(16, 128, 128)`     | -             | -          | Improves training stability           |
| MaxPool2D             | `(16, 64, 64)`       | `2x2 / 2`     | -          | Reduces spatial size                  |
| Conv2D (16 → 32)      | `(32, 64, 64)`       | `3x3 / 1`     | ReLU       | Higher-level features                 |
| BatchNorm2D(32)       | `(32, 64, 64)`       | -             | -          | Prevents internal covariate shift     |
| MaxPool2D             | `(32, 32, 32)`       | `2x2 / 2`     | -          | Further reduces spatial dimensions    |
| Conv2D (32 → 64)      | `(64, 32, 32)`       | `3x3 / 1`     | ReLU       | Deep features                         |
| BatchNorm2D(64)       | `(64, 32, 32)`       | -             | -          | -                                     |
| Dropout (0.6)         | `(64, 32, 32)`       | -             | -          | Prevents overfitting                  |
| Fully Connected       | `(8)`                | -             | -          | Output logits                         |
| Softmax               | `(8)`                | -             | Softmax    | Outputs class probabilities           |

- Convolutional Layers: 4
- Fully connected layers: 1
- Activation: ReLU
- Dropout: 0.6
- Loss Function: Cross Entropy Loss
- Optimizer: Adam (lr=0.001)

### ⚙️ Hyperparameters
| Parameter     | Value   |
|---------------|---------|
| Batch Size    | 32      |
| Learning Rate | 0.001   |
| Epochs        | 30      |
| Weight Decay  | 1e-5    |

### 📊 Results
| Experiment | Epochs | Train Loss | Train Accuracy | Test Accuracy | F1-Score | Avg. Inference Time | Notes |
|------------|--------|------------|----------------|---------------|----------|---------------------|-------|
| Exp-1      | 50     | 0.8006     | 69.08%         | 52.2169%      | 0.4927   | 0.0047 s            | Initial test |

## Model A4
### 🏗️ Model Architecture: WaferCNN_A4
| Layer Type             | Output Shape        | Kernel/Stride | Activation | Notes                                          |
|------------------------|---------------------|---------------|------------|------------------------------------------------|
| Conv2D (1 → 32)         | `(32, 256, 256)`     | `3x3 / 1`     | ReLU       | Initial feature extraction                    |
| BatchNorm2D(32)         | `(32, 256, 256)`     | -             | -          | Normalizes activation outputs                 |
| MaxPool2D               | `(32, 128, 128)`     | `2x2 / 2`     | -          | Downsamples spatial dimensions                |
| Conv2D (32 → 64)        | `(64, 128, 128)`     | `3x3 / 1`     | ReLU       | Learns more complex features                  |
| BatchNorm2D(64)         | `(64, 128, 128)`     | -             | -          | Enhances training stability                   |
| MaxPool2D               | `(64, 64, 64)`       | `2x2 / 2`     | -          | Further reduces spatial dimensions            |
| Conv2D (64 → 128)       | `(128, 64, 64)`      | `3x3 / 1`     | ReLU       | Deep hierarchical feature extraction          |
| BatchNorm2D(128)        | `(128, 64, 64)`      | -             | -          | Mitigates internal covariate shift             |
| MaxPool2D               | `(128, 32, 32)`      | `2x2 / 2`     | -          | Reduces feature map size                      |
| Conv2D (128 → 256)      | `(256, 32, 32)`      | `3x3 / 1`     | ReLU       | Higher-order feature learning                 |
| BatchNorm2D(256)        | `(256, 32, 32)`      | -             | -          | Stabilizes activations                        |
| MaxPool2D               | `(256, 16, 16)`      | `2x2 / 2`     | -          | Further spatial downsampling                  |
| Conv2D (256 → 512)      | `(512, 16, 16)`      | `3x3 / 1`     | ReLU       | Extraction of highly abstract features        |
| BatchNorm2D(512)        | `(512, 16, 16)`      | -             | -          | Regularizes activation statistics             |
| MaxPool2D               | `(512, 8, 8)`        | `2x2 / 2`     | -          | Final pooling prior to global pooling         |
| Squeeze-and-Excitation  | `(512, 8, 8)`        | -             | Sigmoid    | Channel-wise feature recalibration            |
| Global Average Pooling  | `(512, 1, 1)`        | -             | -          | Aggregates spatial information                |
| Flatten                 | `(512)`              | -             | -          | Vectorizes pooled feature maps                |
| Fully Connected (512 → 128) | `(128)`          | -             | ReLU       | Dimensionality reduction                      |
| Dropout                 | `(128)`              | -             | -          | Regularizes and prevents overfitting          |
| Fully Connected (128 → 8) | `(8)`              | -             | -          | Outputs class logits                          |
| Softmax                 | `(8)`                | -             | Softmax    | Outputs class probabilities                   |

- Convolutional Layers: 5
- Fully connected layers: 2
- Activation: ReLU
- Dropout: 0.4
- Loss Function: Focal Loss 
- Optimizer: Adam (lr=0.001)

### ⚙️ Hyperparameters
| Parameter     | Value   |
|---------------|---------|
| Batch Size    | 64      |
| Learning Rate | 0.001   |
| Epochs        | 200     |
| Weight Decay  | 1e-5    |

### 📊 Results
| Experiment | Epochs | Train Loss | Train Accuracy | Test Accuracy | F1-Score | Avg. Inference Time | Notes |
|------------|--------|------------|----------------|---------------|----------|---------------------|-------|
| Exp-1      | 150    | 0.4771     | 73.08%         | 69.7872%      | 0.6865   | 0.0086 s            | Initial test |
| **Exp-2**  | 150    | 0.4514     | 74.31%         | 72.8528%      | 0.7195   | 0.0065 s            |  |
| Exp-3      | 200    | 0.4591     | 73.72%         | 71.8394%      | 0.7099   | 0.0065 s            |  |

## B class notes
Feature extraction:
- Raw pixel flattening 
- Histogram of Oriented Gradients (HOG)
- Gabor Filters (Texture Analysis)

Kernels
- Radial Basis Function (RBF)
- Polynomial
- Sigmoid

Keep in mind:
- Standard feature vectors (zero mean, uit var)
- Dimenionality reduction (PCA?)

## Model B1
### 🏗️ Model Architecture
| Stage             | Input Size             | Output Size           | Transformation      | Notes |
|-------------------|-------------------------|------------------------|---------------------|-------|
| Input Flatten     | `(batch_size, 1, 256, 256)` | `(batch_size, 65536)` | Flatten             | Image flattened to vector |
| RBF Expansion     | `(batch_size, 65536)`    | `(batch_size, 500)`    | exp(-γ‖x-c‖²)        | Maps input into RBF feature space (500 centers) |
| Fully Connected   | `(batch_size, 500)`      | `(batch_size, 8)`      | Linear Transformation | Outputs class scores (no softmax) |

- Feature Mapping: Radial Basis Function (RBF) with γ = 0.0001
- Number of Centers: 500
- Fully Connected Layer: 1
- Activation: Gaussian Kernel in RBF Layer
- Loss Function: Multi-Class Hinge Loss
- Optimizer: Adam (lr=0.01)

### ⚙️ Hyperparameters
| Parameter         | Value   |
|-------------------|---------|
| Batch Size        | 32      |
| Learning Rate     | 0.01    |
| Epochs            | 50      |
| Weight Decay      | 0.001   |
| Scheduler         | StepLR  |
| Step Size         | 30      |
| StepLR Gamma      | 0.1     |
| RBF Gamma (γ)     | 0.0001  |

### 📊 Results
| Experiment | Epochs | Train Loss | Train Accuracy | Test Accuracy | F1-Score | Avg. Inference Time | Notes |
|------------|--------|------------|----------------|---------------|----------|---------------------|-------|
| Exp-1      | 50     | 2.7550     | 56.73%         | 15.1634%      | 0.0764   | 0.0003 s            | Initial test |

### 📝 Observations & Adjustments
- **Exp-1** Pixel flattening for feature extraction not the move

## Model B2
### 🏗️ Model Architecture
| Stage             | Input Size                  | Output Size               | Transformation        | Notes |
|-------------------|-----------------------------|---------------------------|-----------------------|-------|
| Patch Averaging   | `(batch_size, 1, 256, 256)` | `(batch_size, 1, 32, 32)` | AvgPool2D             | Divide into 8x8 patches, average |
| Flatten           | `(batch_size, 1, 32, 32)`   | `(batch_size, 1024)`      | Flatten               | Convert patch map to vector |
| RBF Expansion     | `(batch_size, 1024)`        | `(batch_size, 500)`       | exp(-γ‖x-c‖²)         | Maps input into RBF feature space (500 centers) |
| Fully Connected   | `(batch_size, 500)`         | `(batch_size, 8)`         | Linear Transformation | Outputs class scores (no softmax) |

- Feature Mapping: Radial Basis Function (RBF) with γ = 0.0001
- Number of Centers: 500
- Fully Connected Layer: 1
- Activation: Gaussian Kernel in RBF Layer
- Loss Function: Multi-Class Hinge Loss
- Optimizer: Adam (lr=0.01)

### ⚙️ Hyperparameters
| Parameter         | Value   |
|-------------------|---------|
| Batch Size        | 32      |
| Learning Rate     | 0.01    |
| Epochs            | 50      |
| Weight Decay      | 0.001   |
| Scheduler         | StepLR  |
| Step Size         | 25      |
| StepLR Gamma      | 0.1     |
| RBF Gamma (γ)     | 0.0001  |

### 📊 Results
| Experiment | Epochs | Train Loss | Train Accuracy | Test Accuracy | F1-Score | Avg. Inference Time | Notes |
|------------|--------|------------|----------------|---------------|----------|---------------------|-------|
| Exp-1      | 50     | 2.9893     | 48.43%         | 14.2640%      | 0.0552   | 0.0552 s            | Initial test (inference time on MBP) |

### 📝 Observations & Adjustments
- **Exp-1** sucks...come back to it

## Model C1
### 🏗️ Model Architecture
| Component            | Shape / Purpose                     | Notes                                    |
|----------------------|--------------------------------------|-----------------------------------------|
| Input                | `(1, 256, 256)`                     | Grayscale wafer images                  |
| Flatten              | `(65536,)`                          | Images are flattened before distance computation |
| Training Data Store  | `(num_samples, 65536)`               | Stores flattened training features     |
| Distance Calculation | `(batch_size, num_samples)`          | Compute pairwise distances to training data |
| K Nearest Neighbors  | `(batch_size, k)`                   | Select indices of K closest samples    |
| Voting               | `(batch_size, 8)`                   | Count votes for each class             |
| Output               | `(batch_size, 8)`                   | Raw logits (one per class)              |

### 📊 Results
| Experiment | Test Accuracy | F1-Score | Avg. Inference Time | Notes |
|------------|---------------|----------|---------------------|-------|
| Exp-1      | 35.9387%      | 0.3559   | 3.1748 s            | Euclidean distance, K=5 |
| Exp-2      | 36.4581%      | 0.3516   | 9.9554 s            | Manhattan distance, K=5 |
| Exp-3      | 27.1979%      | 0.2883   | 9.2020 s            | Cosine similarity, K=5 |
| Exp-4      | 36.6101%      | 0.3567   | 10.362 s            | Euclidean distance, K=2 |
| Exp-5      | 35.0266%      | 0.3592   | 3.5632 s            | Euclidean distance, K=1 |
| Exp-6      | 35.6853%      | 0.3521   | 13.901 s            | Euclidean distance, K=7 |
| Exp-7      | 35.0266%      | 0.3592   | 2.9865 s            | Euclidean distance, K=1, inverse distance weighting |

### 📝 Observations & Adjustments
- **Exp-1** High inference time (need to rerun on HPC)
- **Exp-2** Manhattan distance triples runtime, try Cosine
- **Exp-3** Cosine produces worse results, runtime still longer; stick with Euclidean
- **Exp-4** Dropping K helped, new additions to model?
- **Exp-5** Dropping K further didn't help 
- **Exp-6** Increasing K didnt impact much, increase runtime
- **Exp-7** No significant impact

## Model C2
Radius based KNN
### 🏗️ Model Architecture
| Component              | Shape / Purpose                     | Notes                                        |
|-------------------------|--------------------------------------|---------------------------------------------|
| Input                  | `(1, 256, 256)`                     | Grayscale wafer images                     |
| Flatten                | `(65536,)`                          | Images are flattened before distance computation |
| Training Data Store    | `(num_samples, 65536)`               | Stores flattened training features         |
| Distance Calculation   | `(batch_size, num_samples)`          | Compute pairwise distances to training data |
| Radius Selection       | Varies per sample                   | Select all neighbors within a distance radius |
| Fallback Selection     | `(fallback_k,)`                     | Use top-k nearest neighbors if no radius neighbors found |
| Voting                 | `(batch_size, 8)`                   | Count votes for each class                 |
| Output                 | `(batch_size, 8)`                   | Raw logits (one per class)                  |

### 📊 Results
| Experiment | Test Accuracy | F1-Score | Avg. Inference Time | Notes |
|------------|---------------|----------|---------------------|-------|
| Exp-1      | 36.4201%      | 0.3623   | 13.538 s            | r=0.5, K=5, Euclidean |
| Exp-2      | 36.4201%      | 0.3623   | 4.1603 s            | r=1.0, K=5, Euclidean |
| Exp-3      | 36.4201%      | 0.3623   | 18.296 s            | r=2.0, K=5, Euclidean |
| Exp-4      | 00.0000%      | 0.0000   | 0.0000 s            |  |
| Exp-5      | 00.0000%      | 0.0000   | 0.0000 s            |  |

### 📝 Observations & Adjustments
- **Exp-1** Initial test, nothing significantly better than C1
- **Exp-2** Same
- **Exp-3** Same
- **Exp-4** 
- **Exp-5** 
- **Exp-6** 
- **Exp-7** 

