#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets

import argparse
from smdebug import modes
from smdebug.pytorch import get_hook

import logging
import sys
import os
import json

#TODO: Import dependencies for Debugging andd Profiling
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model,criterion,device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    print("testing on whole dataset")
    model.eval()
    running_loss = 0.0
    running_samples = 0.0
    
    batch_size = args.batch_size
    test_batch_size = args.test_batch_size
    _, test_loader  = create_data_loaders(batch_size,test_batch_size)
    
    for inputs,labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        preds = model(inputs)
        
        loss = criterion(preds,labels)
        _, preds = torch.max(preds,1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects = torch.sum(preds == labels.data).item()
        
    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects / len(test_loader.dataset)
        
    logger.info(
    "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
    total_loss, running_corrects, len(test_loader.dataset), 100.0 * running_corrects / len(test_loader.dataset)
        )
            )

def train(model, criterion, optimizer,device):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    epochs = args.epochs
    batch_size = args.batch_size
    test_batch_size = args.test_batch_size
    best_loss = 1e6
    loss_counter = 0
    
    image_dataset, _  = create_data_loaders(batch_size,test_batch_size)
    for epoch in range(epochs):
        for phase in ['train','valid']:
            print(f'epoch {epoch}, phase {phase}')
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
                
            running_loss = 0.0
            running_corrects = 0.0 
            running_samples = 0
        
            for step, (inputs,labels) in enumerate(image_dataset[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)
                loss = criterion(preds,labels)
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _,preds = torch.max(preds,1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                running_samples+=len(inputs)

                if running_samples % 2000 == 0:
                    accuracy = running_corrects / running_samples
                    print("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                                running_samples,
                                len(image_dataset[phase].dataset),
                                100.0 * (running_samples / len(image_dataset[phase].dataset)),
                                loss.item(),
                                running_corrects,
                                running_samples,
                                100.0*accuracy,
                            )
                        )

            epoch_loss = running_loss / running_samples
            epoch_acc  = running_corrects/ running_samples
            
            if phase == "valid":
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    
                else:
                    loss_counter += 1
                    
        if loss_counter == 1:
            break
            
    return model
                
        
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.require_grad = False
        
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features,133))
    
    return model

def create_data_loaders(batch_size,test_batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    
    trainpath = os.path.join(args.data_dir,'train')
    validpath = os.path.join(args.data_dir,'valid')
    testpath = os.path.join(args.data_dir,'test')
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
    ])
    
    trainset = datasets.ImageFolder(trainpath,transform=transform)
    validset = datasets.ImageFolder(validpath,transform=transform)
    testset = datasets.ImageFolder(testpath,transform=transform)
    
    train_loader = torch.utils.data.DataLoader(trainset,shuffle=True,batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(validset,shuffle=True,batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(testset,shuffle=True,batch_size=test_batch_size)
    
    return {"train":train_loader,"valid":valid_loader}, test_loader
    

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")
    
    model = model.to(device)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    model=train(model, loss_criterion, optimizer,device)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, loss_criterion,device)
    
    '''
    TODO: Save the trained model
    '''
    #torch.save(model, path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need

    '''
    
    parser.add_argument("--batch-size",
                        type=int,
                        default=128,
                       )
    parser.add_argument("--epochs",
                        type=int,
                        default=2)
    
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    
    parser.add_argument("--test-batch-size",
                        type=int,
                        default=1000,
                       )
    
    #container Arguments
    
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    
    args=parser.parse_args()
    
    main(args)
