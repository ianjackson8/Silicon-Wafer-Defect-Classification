'''
Defect Detection in Semiconductor Wafers Using Image Classification

Author: Ian Jackson
Version: v1.0

Requirements:
    - WM811K pkl file (see README)

'''

#== Imports ==#
import argparse
import torch
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Union, List
from bcolors import *
from torch.utils.data import DataLoader, Dataset
from torchviz import make_dot
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from torchsummary import summary

#== Global Variables ==#
categories = ['Edge-Ring', 'Center', 'Edge-Loc', 'Loc', 'Random', 'Scratch', 'Donut', 'Near-full']

# CUR_MODEL_PTH = 'saved_models/model_A1-exp9.pth'
CUR_MODEL_PTH = '/scratch/isj0001/Silicon-Wafer-Defect-Classification/saved_models/model_A2-exp5.pth'
CUR_MODEL = 'A2'

training_params = {
    'epochs': 50,
    'dropout': 0.6,
    'lr': 1e-3,
    'weight_decay': 1e-5,
    'scheduler': {
        'use': True,
        'step_size': 5,
        'gamma': 0.5
    }
}

#== Classes ==#
class WaferDataset(Dataset):
    # DOCUMENT: waferdataset class
    def __init__(self, df, transform=None, train=True):
        self.df = df
        self.train = train
        self.transform = transform if transform else self.default_transforms()

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Fx Image Shape: Ensure correct channel-first format (1, 256, 256)
        wafer_img = np.array(row['tensor'])  # Convert to NumPy array
        wafer_img = torch.tensor(wafer_img, dtype=torch.float32)  # Convert to tensor
        # when on HPC, dont need permute?
        # wafer_img = wafer_img.permute(2, 0, 1)  # Convert from (H, W, C) -> (C, H, W)
        
        # Fx Labels: Convert from one-hot encoding to class indices
        label = torch.tensor(row['failureTypeVector'], dtype=torch.float32)  # Convert to tensor
        label = label.argmax().long()  # Convert one-hot vector to class index

        # augmentation 
        if self.transform and self.train:
            wafer_img = self.transform(wafer_img)

        return wafer_img, label
    
    def default_transforms(self):
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(90)
        ])

class WaferCNN_A1(nn.Module):
    def __init__(self, num_classes:int=8):
        '''
        initialization of CNN for Wafer Classification
        Model A1

        Args:
            num_classes (int, optional): number of classes. Defaults to 8.
        '''
        super(WaferCNN_A1, self).__init__()

        #= Block 1 =#
        # Input: 1×256×256
        # Output: 32×256×256  (conv), then 32×128×128 (pool)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        
        #= Block 2 =#
        # Input: 32×128×128
        # Output: 64×128×128 (conv), then 64×64×64 (pool)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        
        #= Block 3 =#
        # Input: 64×64×64
        # Output: 128×64×64 (conv), then 128×32×32 (pool)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        
        #= Block 4 =#
        # Input: 128×32×32
        # Output: 256×32×32 (conv), then 256×16×16 (pool)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(256)
        
        # After the 4th block, we expect feature maps of size 256×16×16 so the flattened size is 256 * 16 * 16 = 65536
        
        #= Fully Connected Layers =#
        # You can reduce the dimension with one or two Dense layers
        self.fc1 = nn.Linear(256 * 16 * 16, 256)   # from flattened feature map to 256
        self.fc2 = nn.Linear(256, num_classes)     # final classification to 8 classes

        # Optionally, a dropout layer can be added in between fc1 and fc2:
        self.dropout = nn.Dropout(training_params['dropout'])

    def forward(self, x):
        #= Block 1 =#
        # conv -> BatchNorm -> ReLU -> pool
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        #= Block 2 =#
        # conv -> BatchNorm -> ReLU -> pool
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        #= Block 3 =#
        # conv -> BatchNorm -> ReLU -> pool
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)

        #= Block 4 =#
        # conv -> BatchNorm -> ReLU -> pool
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)

        # Flatten
        x = x.reshape(x.size(0), -1)

        # fully connect layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class WaferCNN_A2(nn.Module):
    def __init__(self, num_classes:int=8):
        '''
        initialization of CNN for Wafer Classification
        Model A1

        Args:
            num_classes (int, optional): number of classes. Defaults to 8.
        '''
        super(WaferCNN_A2, self).__init__()

        #= Block 1 =#
        # Input: 1×256×256
        # Output: 32×256×256  (conv), then 32×85×85 (pool)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        
        #= Block 2 =#
        # Input: 32×85×85
        # Output: 64×85×85 (conv), then 64×42×42 (pool)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        
        #= Block 3 =#
        # Input: 64×42×42
        # Output: 128×42×42 (conv), then 128×21×21 (pool)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        
        # After the 3rd block, we expect feature maps of size 128×21×21 so the flattened size is 128 * 21 * 21 = 56488
        
        #= Fully Connected Layers =#
        # You can reduce the dimension with one or two Dense layers
        self.fc1 = nn.Linear(128 * 21 * 21, 128)   # from flattened feature map to 128
        self.fc2 = nn.Linear(128, num_classes)     # final classification to 8 classes

        # Optionally, a dropout layer can be added in between fc1 and fc2:
        self.dropout = nn.Dropout(training_params['dropout'])

    def forward(self, x):
        #= Block 1 =#
        # conv -> BatchNorm -> ReLU -> pool
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 3)

        #= Block 2 =#
        # conv -> BatchNorm -> ReLU -> pool
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        #= Block 3 =#
        # conv -> BatchNorm -> ReLU -> pool
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)

        # Flatten
        x = x.reshape(x.size(0), -1)

        # fully connect layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

#== Methods ==#
def load_dataset(loc: str, calc_type_counts: bool) -> Union[pd.DataFrame, pd.DataFrame]:
    '''
    Load the WM811K dataset from pickle file
    Cleans and returns test and training set as Pandas Dataframe

    Args:
        loc (str): location of pkl file
        calc_type_counts (bool): if true, calc type counts from dataset 

    Returns:
        Union[pd.DataFrame, pd.DataFrame]: (train df, test df)
    '''
    print("[i] Importing WK-811K dataset")

    # read pickle file
    df = pd.read_pickle(loc)

    # filter out non-test/training entries
    # df = df[~df["trainTestLabel"].apply(lambda x: isinstance(x, np.ndarray) and np.array_equal(x, np.array([0, 0])))]

    # if the label is [0,0] consider it training
    df["trainTestLabel"] = df["trainTestLabel"].apply(
        lambda x: "Training" if isinstance(x, np.ndarray) and np.array_equal(x, np.array([0, 0])) else x
    )

    # remove entries with the none and [0,0] failure type label
    df = df[~df['failureType'].apply(lambda x: isinstance(x, str) and x == 'none')]
    df = df[~df['failureType'].apply(lambda x: isinstance(x, np.ndarray) and np.array_equal(x, np.array([0, 0])))]

    # one-hot vector the failure types
    cat_to_index = {cat: i for i, cat in enumerate(categories)}

    def one_hot_encode(cat):
        vector = [0] * len(categories)
        vector[cat_to_index[cat]] = 1
        return vector
    
    df['failureTypeVector'] = df['failureType'].apply(one_hot_encode)

    # create a column of the wafer sizes
    df['rows'] = df['waferMap'].apply(lambda x: len(x))
    df['cols'] = df['waferMap'].apply(lambda x: len(x[0]) if len(x) > 0 else 0)

    # MAX -> (212, 204)
    # max_x = df['rows'].max()
    # max_y = df['cols'].max()
    # print(f"Max - ({max_x}, {max_y})") 

    # separate test and train
    train_df = df[df['trainTestLabel'] == 'Training']
    test_df  = df[df['trainTestLabel'] == 'Test']

    # reindex df
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # get counts of labels for training set
    if calc_type_counts:
        train_failure_type_counts = train_df["failureType"].value_counts().to_dict()
        test_failure_type_counts = test_df["failureType"].value_counts().to_dict()
    else: 
        train_failure_type_counts = None
        test_failure_type_counts = None

    # print message
    print("[i] Loaded WK-811K dataset")
    print(f"\tTraining Size: {len(train_df)}")
    if calc_type_counts: print(f"\t\t{train_failure_type_counts}")
    print(f"\tTest Size:     {len(test_df)}")
    if calc_type_counts: print(f"\t\t{test_failure_type_counts}")

    return train_df, test_df

def pad_wafer_map(wafer_map_2d: List[List], target_row=256, target_col=256) -> np.array:
    '''
    pad the 2D wafer map and convert to 3D numpy array of shape (target_row, target_col, 1)

    Args:
        wafer_map_2d (List[List]): 2D wafer map
        target_row (int, optional): number of rows to pad to. Defaults to 256.
        target_cols (int, optional): number of cols to pad to. Defaults to 256.

    Returns:
        np.array: 3D NumPy array
    '''
    # convert to NumPy array, get cur dimensions
    arr = np.array(wafer_map_2d)
    r,c = arr.shape

    # create target sized array of zeros
    padded_arr = np.zeros((target_row, target_col), dtype=arr.dtype)

    # copy to top-left corner
    padded_arr[:r, :c] = arr

    # add a channel dimension 
    padded_arr = np.expand_dims(padded_arr, axis=0)

    return padded_arr

#== Main Execution ==#
def main(args):
    print("== Defect Detection in Semiconductor Wafers Using Image Classification ==")

    #-- STEP 1: Load Dataset --#
    # check if dataset has been loaded already
    if not (os.path.exists('Clean_Train_WM811K.pkl') and os.path.exists('Clean_Test_WM811K.pkl')) or args.force_load_dataset:
        # load the WM-811K dataset
        data_loc = "WM811K.pkl"
        train_df, test_df = load_dataset(data_loc, args.calc_type_counts)

        # convert each waferMap into tensor
        print("[i] Padding wafer map data")
        train_df['tensor'] = train_df['waferMap'].apply(lambda wm: pad_wafer_map(wm))
        test_df['tensor'] = test_df['waferMap'].apply(lambda wm: pad_wafer_map(wm))

        # save dataset
        print("[i] Saving dataset to file")
        train_df.to_pickle('Clean_Train_WM811K.pkl')
        test_df.to_pickle('Clean_Test_WM811K.pkl')

    else:
        # load dataset if already exist
        print("[i] Loading dataset from file")
        train_df = pd.read_pickle('Clean_Train_WM811K.pkl')
        test_df = pd.read_pickle('Clean_Test_WM811K.pkl')
    
    #-- STEP 2: Prepare Model --#
    # initialize the model
    print("[i] Initializing model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Using device: {device}")

    # determine model and init it
    if CUR_MODEL == 'A1': model = WaferCNN_A1(num_classes=8).to(device)
    elif CUR_MODEL == 'A2': model = WaferCNN_A2(num_classes=8).to(device)
    else: 
        print(f'[E] Model {CUR_MODEL} not defined')
        quit()

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=training_params['lr'], weight_decay=training_params['weight_decay'])

    # define scheduler (Reduce LR by 50% every 10 epochs)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=training_params['scheduler']['step_size'], gamma=training_params['scheduler']['gamma'])

    # prepare DataLoader
    print("[i] Prepare dataloader")
    train_dataset = WaferDataset(df=train_df, train=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    # visualize model 
    if args.visualize:
        print("[i] Visualizing model")
        summary(model, input_size=(1, 256, 256))
        x = torch.randn(1, 1, 256, 256)
        y = model(x)
        make_dot(y, params=dict(model.named_parameters())).render("model_A1", format="png")
        quit()

    #-- STEP 3: Train Model --#
    if not os.path.exists(CUR_MODEL_PTH) or args.force:
        # training loop
        print("[i] Beginning training loop")
        start_time = time.time()
        for epoch in range(training_params['epochs']):
            model.train()
            running_loss, correct, total = 0.0, 0, 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)

                # zero gradients
                optimizer.zero_grad()

                # forward
                outputs = model(images)

                # compute loss
                loss = criterion(outputs, labels)

                # backward prop
                loss.backward()

                # update params 
                optimizer.step()

                # accumulate stats
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / total
            train_acc = correct / total

            # step scheduler
            if training_params['scheduler']['use']: scheduler.step()

            print(f"Epoch [{epoch+1}/{training_params['epochs']}], "
                f"\tLoss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
    
        # save the model
        torch.save(model.state_dict(), CUR_MODEL_PTH)
        print("[i] Model training complete and saved.")
        model.eval()

        # train time
        end_time = time.time()
        total_time = end_time - start_time
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"[i] Total training time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")

    # model already trained, load
    else:
        print("[i] Loading model from file")
        model.load_state_dict(torch.load(CUR_MODEL_PTH))
        model.eval()

    # run the model on the test set
    print("[i] Running model on test set")
    test_dataset = WaferDataset(df=test_df, train=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # measure inference time
            start_time = time.time()
            outputs = model(images)
            end_time = time.time()

            inference_time = end_time - start_time

            # get predicted classes
            _, pred = torch.max(outputs, 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # calculate accuracy and f1 score
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    avg_inference_time = np.mean(inference_time)

    # compute confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # per-class accuracy
    class_report = classification_report(all_labels, all_preds, target_names=categories)

    # print results
    print(f"\tAccuracy: {accuracy*100:.4f}%")
    print(f"\tF1 Score: {f1:.4f}")
    print(f"\tAvg Inference Time/batch: {avg_inference_time:.4f} seconds")
    print("\tPer-Class Accuracy Report:")
    print(class_report)

    # plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=categories, yticklabels=categories)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")

if __name__ == "__main__":
    # argparse 
    parser = argparse.ArgumentParser()

    parser.add_argument('--calc_type_counts', action='store_true', help='Calculates and prints failure type counts from the dataset. Increases load time.')
    parser.add_argument('--force_load_dataset', action='store_true', help='Forces the dataset to be loaded even if it has been loaded already.')
    parser.add_argument('--force', action='store_true', help='Forces the model to be trained even if it has been trained already.')
    parser.add_argument('--visualize', action='store_true', help='Visualizes the model using torchviz.')

    args = parser.parse_args()

    main(args)