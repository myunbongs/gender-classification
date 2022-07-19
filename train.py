import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import time
import copy
from torchvision import models

import argparse

from data import make_celeba_dataloaders


def get_parser():
    parser = argparse.ArgumentParser(description="transfer learing assignment")
    parser.add_argument("--path", type=str, default='celeba-gender-dataset/',
                        help="Path to the directory to dataset.")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of classes.")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Number of epochs.")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate.")
    parser.add_argument("--save_dir", type=str, default='checkpoints/model.pt',
                        help="Path to the directory to save model.")
    return parser

class Trainer(): 
    def __init__(self, epochs, learning_rate, dataloaders, device, dataset_sizes, save_dir, num_classes):

        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, self.num_classes)

        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=learning_rate)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

        self.dataloaders = dataloaders
        self.epochs = epochs
        self.device = device
        self.dataset_sizes = dataset_sizes
        self.save_dir = save_dir

    def train(self):

        since = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch}/{self.num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    self.scheduler.step()

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        torch.save(self.model, self.save_dir)

        return self.model

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = get_parser()
    args = parser.parse_args()
    
    celeba_dataloaders, dataset_sizes = make_celeba_dataloaders(path=args.path)

    trainer = Trainer(epochs=args.epochs, learning_rate=args.learning_rate, dataloaders=celeba_dataloaders, device=device, dataset_sizes=dataset_sizes, save_dir=args.save_dir, num_classes=args.num_classes)
    trainer.train()

if __name__ == '__main__':
    main()


