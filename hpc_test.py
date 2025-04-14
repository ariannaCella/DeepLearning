import torch
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from solver import Solver
from dataset.dataset import load_dataset, load_dataset_6535 # load_dataset_70_30,
from utils import plot_confusion_matrix, plot_confusion_matrix_with_errors, analyze_class_performance


def get_args():
    parser = argparse.ArgumentParser()   

    parser.add_argument('--run_name', type=str, default="run_1", help='name of current run')
    parser.add_argument('--model_name', type=str, default="first_train", help='name of the model to be saved/loaded')

    parser.add_argument('--epochs', type=int, default=2, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='number of elements in batch size')
    parser.add_argument('--workers', type=int, default=2, help='number of workers in data loader')
    parser.add_argument('--print_every', type=int, default=500, help='print losses every N iteration')

    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--opt', type=str, default='SGD', choices=['SGD', 'Adam'], help = 'optimizer used for training')
    parser.add_argument('--use_norm', action='store_true', help='use normalization layers in model')
    parser.add_argument('--feat', type=int, default=16, help='number of features in model')

    parser.add_argument('--dataset_path', type=str, default='./data', help='path were to save/get the dataset')
    parser.add_argument('--checkpoint_path', type=str, default='./', help='path were to save the trained model')

    parser.add_argument('--resume_train', action='store_true', help='load the model from checkpoint before training')

    return parser.parse_args()



def main(args):
    writer = SummaryWriter('./runs/' + args.run_name)

    #trainset, valset, testset, class_names = load_dataset(args.dataset_path)
    #trainset, valset, testset, class_names = load_dataset_70_30(args.dataset_path)
    trainset, valset, testset, class_names = load_dataset_6535(args.dataset_path)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    validationloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    # define solver class
    solver = Solver(train_loader=trainloader,
            validation_loader=validationloader,
            test_loader=testloader,
            classes=class_names,
            device=device,
            writer=writer,
            args=args)

    # TRAIN model
    solver.train()

    # Test Model
    all_preds, all_labels = solver.test()

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plot_confusion_matrix(cm, class_names, writer=writer, global_step=args.epochs)
    plot_confusion_matrix_with_errors(cm, class_names, writer=writer, global_step=args.epochs)
    analyze_class_performance(all_labels, all_preds, cm, class_names, top_n=5)
    


if __name__ == "__main__":
    args = get_args()
    print(args)
    main(args)