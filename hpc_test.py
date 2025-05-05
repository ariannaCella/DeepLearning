import torch
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time

from solver import Solver
from dataset.dataset import load_dataset, load_dataset_50_50
from utils import plot_confusion_matrix, plot_confusion_matrix_with_errors, analyze_class_performance

import random

# setto un seed fisso (da scegliere) per rendere i risultati riproducibili
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id,seed):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)


def get_args():
    parser = argparse.ArgumentParser()   

    parser.add_argument('--run_name', type=str, default="run_1", help='name of current run')
    parser.add_argument('--model_name', type=str, default="first_train", help='name of the model to be saved/loaded')

    parser.add_argument('--epochs', type=int, default=2, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='number of elements in batch size')
    parser.add_argument('--workers', type=int, default=2, help='number of workers in data loader')
    parser.add_argument('--print_every', type=int, default=500, help='print losses every N iteration')
    parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')
    parser.add_argument('--aug', action='store_true', help='Enable data augmentation')


    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--opt', type=str, default='SGD', choices=['SGD', 'Adam'], help = 'optimizer used for training')
    parser.add_argument('--use_norm', action='store_true', help='use normalization layers in model')
    parser.add_argument('--feat', type=int, default=16, help='number of features in model')

    parser.add_argument('--dataset_path', type=str, default='./data', help='path were to save/get the dataset')
    parser.add_argument('--checkpoint_path', type=str, default='./', help='path were to save the trained model')

    parser.add_argument('--resume_train', action='store_true', help='load the model from checkpoint before training')

    return parser.parse_args()



def main(args):
    set_seed(args.seed)
    # Generatore e worker seed
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    writer = SummaryWriter('./runs/' + args.run_name)

    trainset, valset, testset, class_names = load_dataset(args.dataset_path,args.seed,args.aug)
    #trainset, valset, testset, class_names = load_dataset_50_50(args.dataset_path,args.seed,args.aug)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, worker_init_fn=lambda worker_id: seed_worker(worker_id, args.seed), generator=g)
    validationloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, worker_init_fn=lambda worker_id: seed_worker(worker_id, args.seed), generator=g)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, worker_init_fn=lambda worker_id: seed_worker(worker_id, args.seed), generator=g)

    
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
    start_time = time.time()
    solver.train()
    # Tempo di addestramento
    elapsed = int(time.time() - start_time)
    h, m = divmod(elapsed // 60, 60)
    s = elapsed % 60
    training_time_str = f"{h}h {m}m {s}s"

    # Test Model
    all_preds, all_labels = solver.test()

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Analisi dei risultati
    plot_confusion_matrix(cm, class_names, writer=writer, global_step=args.epochs)
    plot_confusion_matrix_with_errors(cm, class_names, writer=writer, global_step=args.epochs)
    analyze_class_performance(all_labels, all_preds, cm, class_names, writer, top_n=5)
    

    # Logga tabella Markdown per recap dell'esecuzione
    markdown_table = f"""
| Metric                    | Value                         |
|---------------------------|-------------------------------|
| Model Name                | {args.model_name}             |
| Net Name                  | {solver.net_model_name}       |
| Batch Size                | {args.batch_size}             |
| Learning Rate             | {args.lr}                     |
| Optimizer                 | {args.opt}                    |
| Epochs                    | {args.epochs}                 |
| Patience                  | {solver.patience}             |
| Best Validation Accuracy  | {solver.best_accuracy:.2f}%   |
| Final Test Accuracy       | {solver.test_accuracy:.2f}%   |
| Training Time             | {training_time_str}           |
"""
    writer.add_text("Experiment Summary", markdown_table, global_step=args.epochs)
    writer.close()
    


if __name__ == "__main__":
    args = get_args()
    print(args)
    main(args)
