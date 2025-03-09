- [Model A2](#model-a2)
  - [Experiment 1](#experiment-1)
# Model A2

## Experiment 1
| Epoch | Loss   | Accuracy |
|-------|--------|----------|
| 1     | 1.0572 | 0.6546   |
| 2     | 0.7643 | 0.7090   |
| 3     | 0.6602 | 0.7470   |
| 4     | 0.5568 | 0.7922   |
| 5     | 0.4841 | 0.8196   |
| 6     | 0.4266 | 0.8405   |
| 7     | 0.4002 | 0.8506   |
| 8     | 0.3672 | 0.8644   |
| 9     | 0.3296 | 0.8779   |
| 10    | 0.3042 | 0.8907   |

**Per class accuracy report**
|    class     |precision |   recall|  f1-score  | support|
|--------------|----------|---------|------------|--------|
|  Edge-Ring   |   0.84   |  0.88   |  0.86      | 1126   |
|     Center   |   0.62   |  0.83   |  0.71      |  832   |
|   Edge-Loc   |   0.64   |  0.87   |  0.74      | 2772   |
|        Loc   |   0.56   |  0.43   |  0.49      | 1973   |
|     Random   |   0.66   |  0.71   |  0.68      |  257   |
|    Scratch   |   0.48   |  0.04   |  0.07      |  693   |
|      Donut   |   0.67   |  0.01   |  0.03      |  146   |
|  Near-full   |   0.00   |  0.00   |  0.00      |   95   |
|              |          |         |            |        |    
|    accuracy  |          |         |   0.65     | 7894   |
|   macro avg  |    0.56  |   0.47  |   0.45     | 7894   |
|weighted avg  |    0.63  |   0.65  |   0.61     | 7894   |

**Confusion Matrix**
![exp1-confusion_matrix](./exp1-confusion_matrix.png)
