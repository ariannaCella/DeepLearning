import torch
import torch.optim as optim
import torchvision
import torch.nn as nn
import os
import numpy as np

from model import get_model

class Solver(object):
    """Solver for training and testing."""

    def __init__(self, train_loader, validation_loader, test_loader, classes, device, writer, args):
        """Initialize configurations."""

        self.args = args
        self.model_name = 'stanford_net_{}.pth'.format(self.args.model_name)
        self.classes_name=classes
        self.epochs = self.args.epochs
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.device = device
        self.writer = writer

        # Early stopping
        self.patience = 8  # Numero massimo di epoche senza miglioramenti
        self.best_accuracy = 0
        self.early_stop_counter = 0
        
        # Seleziona modello pre addestrato da usare 
        model_name = "resnet18"
        self.net = get_model(model_name, num_classes=196)
        self.net = self.net.to(device)  
        
        # load a pretrained model
        if self.args.resume_train == True:
            self.load_model()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Choose optimizer
        if self.args.opt == "SGD":
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=0.9)
        elif self.args.opt == "Adam":
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr)


    def save_model(self):
        # if you want to save the model
        check_path = os.path.join(self.args.checkpoint_path, self.model_name)
        torch.save(self.net.state_dict(), check_path)
        print("Model saved!")


    def load_model(self):
        # function to load the model
        check_path = os.path.join(self.args.checkpoint_path, self.model_name)
        self.net.load_state_dict(torch.load(check_path))
        print("Model loaded!")


    def train(self):
        self.net.train()
        for epoch in range(self.epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                
                # put data on correct device
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                
            print(f'[{epoch + 1}] Training loss: {running_loss / len(self.train_loader):.3f}')
            self.writer.add_scalar('Training loss',
                running_loss / len(self.train_loader),
                epoch)

            # Valuate model ed early stopping
            val_accuracy = self.validation(epoch)
            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                self.early_stop_counter = 0
                self.save_model()
            else:
                self.early_stop_counter += 1
                print(f"Early stopping counter: {self.early_stop_counter}/{self.patience}")
                if self.early_stop_counter >= self.patience:
                    print("Early stopping triggered. Stopping training.")
                    break
        
        self.writer.flush()
        self.writer.close()
        print('Finished Training')   
    

    def validation(self, epoch):
        # now lets evaluate the model on the validation set
        correct = 0
        total = 0
        val_loss = 0
        # put net into evaluation mode
        self.net.eval()

        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in self.validation_loader:
                inputs, labels = data
                # put data on correct device
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # calculate outputs by running images through the network
                outputs = self.net(inputs)

                loss = self.criterion(outputs, labels)
                # print statistics
                val_loss += loss.item()

                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        self.writer.add_scalar('Validation accuracy',
            val_accuracy,
            epoch)
        self.writer.add_scalar('Validation loss',
            val_loss / len(self.validation_loader),
            epoch)

        print(f'[{epoch + 1}] Validation loss: {val_loss / len(self.validation_loader):.3f}')
        print(f'Accuracy of the network on the validation images: {val_accuracy} %')

        self.net.train()
        return val_accuracy


    def test(self):
        # now lets evaluate the model on the test set
        correct = 0
        total = 0
        
        # load best model saved
        self.load_model()
        # put net into evaluation mode
        self.net.eval()

        # save pred e labels
        all_preds = []
        all_labels = []

        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data
                # put data on correct device
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # calculate outputs by running images through the network
                outputs = self.net(inputs)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        self.writer.add_scalar('Test accuracy',
            100 * correct / total)

        print(f'Accuracy of the network on the test images: {100 * correct / total} %')
        
        return all_preds, all_labels
        
        
