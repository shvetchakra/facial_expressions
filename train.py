import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms, models
from PIL import Image

from model import EmotionClassifier

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    
    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionClassifier()

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    

    print("Done loading model.")
    return model

def _get_train_and_valid_data_loader(batchSize, train_dir, valid_dir):
    print("Get train and valid data loader.")

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomResizedCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    


    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batchSize, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=batchSize, shuffle=False)
    return trainloader, validloader


def train(model, train_loader, valid_loader, epochs, optimizer, loss_fn, device):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    optimizer    - The optimizer to use during training.
    loss_fn      - The loss function used for training.
    device       - Where the model and data should be loaded (gpu or cpu).
    """
    prev_acc = 0
    accuracy = 0
    softmax = nn.LogSoftmax(dim=1)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:         
            batch_X, batch_y = batch
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)         

            optimizer.zero_grad()
            logps = model.forward(batch_X)
            loss = loss_fn(softmax(logps), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.data.item()
            
        print("Epoch: {}, NLLLoss: {}".format(epoch, total_loss / len(train_loader)))
        model.eval()
        valid_loss = 0
        accuracy = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)
                batch_loss = loss_fn(softmax(logps), labels)

                valid_loss += batch_loss.item()

                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                valid_acc = accuracy/len(valid_loader)

        print(f"Epoch {epoch+1}/{epochs}.. "
      f"Train loss: {total_loss / len(train_loader):.3f}.. "
      f"Valid loss: {valid_loss/len(valid_loader):.3f}.. "
      f"Validation accuracy: {valid_acc:.3f}")
        
        model.train()
        if valid_loss-total_loss > .1 and valid_acc > .62:
            break
        else:
            prev_acc = valid_acc

if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments when the script
    # is executed. Here we set up an argument parser to easily access the parameters.

    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument('--batch-size', type=int, default=96, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # Model Parameters
    parser.add_argument('--learning_rate', type=float, default=0.001, metavar='N',
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--resume_training', type=bool, default=False,
                        help='If you like to resume the training')

    # SageMaker Parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Load the training data.
    train_dir = os.path.join(args.data_dir, "train")
    valid_dir = os.path.join(args.data_dir, "valid")
    train_loader , valid_loader = _get_train_and_valid_data_loader(args.batch_size, train_dir, valid_dir)

    # Build the model.
    model = None
    model_dir=args.data_dir
    print("Model Dir: {}".format(model_dir))
    if args.resume_training:
        model = model_fn(model_dir)
    else:
        model = EmotionClassifier()
    model = model.to(device)
    """print("Model loaded with hidden_dim {} ".format(
        args.embedding_dim, args.hidden_dim, args.vocab_size
    ))"""

    # Train the model.
    optimizer = optim.Adam(model.parameters(), args.learning_rate)
    loss_fn = torch.nn.NLLLoss()

    train(model, train_loader, valid_loader, args.epochs, optimizer, loss_fn, device)

    

	# Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
