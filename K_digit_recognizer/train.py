import torch
import os
import numpy as np
from torchvision import transforms
from model import resnet34
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
from construct_myDataSet import csv_Dataset
import random
import torch.utils.data




      




def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    #--------------------- construct dataset,train_dataset,validate_dataset --------------------------------------------
    data_transform =    {
                            "train":transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]), # 参数来自pytorch官方
                            "val":transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                        }
    dataset = csv_Dataset(csv_path = "/wx/Models_WX/DataSets/digit_recognizer/train.csv", label_idx=0, transform=None)
    num_dataset = len(dataset)
    train_dataset,validate_dataset = torch.utils.data.random_split(dataset,[int(num_dataset*0.9),int(num_dataset*0.1)])
    train_num = len(train_dataset)
    val_num = len(validate_dataset)

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                batch_size=batch_size, shuffle=False,
                                                num_workers=nw)                            
    print("using {} images for training, {} images for validation.".format(train_num,val_num))

    #--------------------------- training: using ResNet34 -------------------------------------------------------
    net = resnet34(num_classes=10)
    model_weight_path = "/wx/Models_WX/K-T/K_digit_recognizer/resnet34_pre.pth" # Transfer Learning
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    # change fc layer structure (0-9)
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 10)
    net.to(device)
    # define loss function
    loss_function = nn.CrossEntropyLoss()
    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)
    # training
    epochs = 3
    best_acc = 0.0
    save_path = '/wx/Models_WX/K-T/K_digit_recognizer/resnet34_digit_recognizer.pth'
    train_steps = len(train_loader) 
    print(train_steps)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for _, data in enumerate(train_bar):
            # images, labels = data
            # print(len(images))
            images, labels = data[0].type(torch.cuda.FloatTensor).to(device), data[1].to(device)
            # print(images.dtype)
            optimizer.zero_grad()
            logits = net(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,epochs,loss)
        # validation
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                val_images = val_images.type(torch.cuda.FloatTensor)
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,epochs)
        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')

if __name__ == '__main__':
    main()