import torch
import pandas as pd
import os
import numpy as np






class csv_Dataset(torch.utils.data.Dataset):
    def __init__(self,csv_path,label_idx=-1,transform=None):
        # self.csv_path = csv_path
        # self.label_idex = label_idx
        self.transform = transform

        assert os.path.exists(csv_path), "{} path does not exist.".format(csv_path)
        df_train = pd.read_csv(csv_path)
        y_train = df_train[df_train.columns[label_idx]]
        X_train = df_train.drop(columns = df_train.columns[label_idx],axis=1)

        dataset = []
        # Series -> np.array -> tensor
        for row_index,row in X_train.iterrows():
            np_img_flatten = np.array(row.tolist())
            np_img = np_img_flatten.reshape(28,28)
            tensor_img = torch.tensor(np_img)
            input_tensor_img = torch.unsqueeze(tensor_img,0) # torch.Size([28, 28, 1])
            label = y_train[row_index]
            train_data = (input_tensor_img,label)
            dataset.append(train_data)
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        image, label = self.dataset[idx]
        return image, label 


 