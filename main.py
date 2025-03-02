'''
Defect Detection in Semiconductor Wafers Using Image Classification

Author: Ian Jackson
Version: v0.1

Requirements:
    - WM811K pkl file (see README)

'''

#== Imports ==#
import argparse
import torch
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from typing import Union, List
from bcolors import *
from torch.utils.data import DataLoader, Dataset

#== Global Variables ==#
NUM_EPOCHS = 10

categories = ['Edge-Ring', 'Center', 'Edge-Loc', 'Loc', 'Random', 'Scratch', 'Donut', 'Near-full']

#== Classes ==#
class WaferDataset(Dataset):
    # DOCUMENT: waferdataset class
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Fx Image Shape: Ensure correct channel-first format (1, 256, 256)
        wafer_img = np.array(row['tensor'])  # Convert to NumPy array
        wafer_img = torch.tensor(wafer_img, dtype=torch.float32)  # Convert to tensor
        wafer_img = wafer_img.permute(2, 0, 1)  # Convert from (H, W, C) -> (C, H, W)
        
        # Fx Labels: Convert from one-hot encoding to class indices
        label = torch.tensor(row['failureTypeVector'], dtype=torch.float32)  # Convert to tensor
        label = label.argmax().long()  # Convert one-hot vector to class index

        return wafer_img, label

class WaferCNN(nn.Module):
    def __init__(self, num_classes:int=8):
        '''
        initialization of CNN for Wafer Classification

        Args:
            num_classes (int, optional): number of classes. Defaults to 8.
        '''
        super(WaferCNN, self).__init__()

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
        self.dropout = nn.Dropout(0.5)

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

def pad_wafer_map(wafer_map_2d: List[List], target_row=256, target_cols=256) -> np.array:
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
    padded_arr = np.zeros((target_row, target_cols), dtype=arr.dtype)

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
    model = WaferCNN(num_classes=8)

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # prepare DataLoader
    print("[i] Prepare dataloader")
    train_dataset = WaferDataset(df=train_df)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # DEBUGGING
    if False:
        print("[D] Sample Training Data")
        print("[D] train dataframe information")
        print(train_df.info())

        print("[D] train dataframe sample id=0")
        sample = train_df.iloc[0]
        print(f'\tdieSize: {sample["dieSize"]}')
        print(f'\tfaultType: {sample["failureType"]}')
        print(f'\tlotName: {sample["lotName"]}')
        print(f'\trainTestLabel: {sample["trainTestLabel"]}')
        print(f'\twaferIndex: {sample["waferIndex"]}')
        print(f'\twaferMap: {sample["waferMap"]}')
        print(f'\tfailureTypeVector: {sample["failureTypeVector"]}')
        print(f'\trows: {sample["rows"]}')
        print(f'\tcols: {sample["cols"]}')
        print(f'\ttensor: {sample["tensor"].shape}')

        wafer_2d = sample['tensor'].squeeze()
        plt.imshow(wafer_2d, cmap='gray')  # Use grayscale for better visualization
        plt.colorbar()  # Show color scale
        plt.title("Wafer Map")
        plt.show()
        quit()

    #-- STEP 3: Train Model --#
    # training loop
    print("[i] Beginning training loop")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for batch_idx, (images, labels) in enumerate(train_loader):
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

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], "
              f"\tLoss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        
    # save the model
    torch.save(model.state_dict(), 'waferCNN_model.pth')
    print("Model training complete and saved.")

if __name__ == "__main__":
    # argparse 
    parser = argparse.ArgumentParser()

    parser.add_argument('--calc_type_counts', action='store_true', help='Calculates and prints failure type counts from the dataset. Increases load time.')
    parser.add_argument('--force_load_dataset', action='store_true', help='Forces the dataset to be loaded even if it has been loaded already.')

    args = parser.parse_args()

    main(args)