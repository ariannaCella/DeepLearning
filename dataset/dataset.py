import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

def get_transforms():
    return {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), shear=10),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test_val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }


def load_dataset(dataset_path):
    data_transforms = get_transforms()
    # load train ds
    full_trainset = torchvision.datasets.StanfordCars(root=dataset_path, split='train',
                                                 download=False, transform=None)
    class_names=full_trainset.classes
    
    # load test set
    testset = torchvision.datasets.StanfordCars(root=dataset_path, split='test',
                                                download=False, transform=data_transforms['test_val'])
                                                
    class_names=full_trainset.classes
    labels = np.array([label for _, label in full_trainset._samples])

    # train set in 70% training e 30% validation mantenendo le proporzioni delle classi
    train_indices, val_indices = train_test_split(np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=42)

    
    trainset = torch.utils.data.Subset(torchvision.datasets.StanfordCars(root=dataset_path, split='train', download=False, transform=data_transforms['train']), train_indices)
    valset = torch.utils.data.Subset(torchvision.datasets.StanfordCars(root=dataset_path, split='train', download=False, transform=data_transforms['test_val']), val_indices)

    print(f"Train set size: {len(trainset)}, Validation set size: {len(valset)}, Test set size: {len(testset)}")

    return trainset, valset, testset, class_names
    


def load_dataset_6535(dataset_path):
    data_transforms = get_transforms()
    
    # load train ds 
    trainset = torchvision.datasets.StanfordCars(root=dataset_path, split='train',
                                                  download=False, transform=data_transforms['train'])
    class_names=trainset.classes
    
    # load test set
    full_test = torchvision.datasets.StanfordCars(root=dataset_path, split='test',
                                                download=False, transform=data_transforms['test_val'])
                                                
    labels = np.array([label for _, label in full_test._samples])
    
    # test set in 70% test e 30% validation mantenendo le proporzioni delle classi
    test_indices, val_indices = train_test_split(np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=42)
    #0.3  trainset (t+v)=66% e test set 34% e abbiamo 77% train e 23% per val (sul training)
    #0.2  trainset (t+v) =                test 6432 t+v 9752   (suddivisione 60:40 test e training)
    #0.4  70:30
    
    testset = torch.utils.data.Subset(full_test, test_indices)
    valset = torch.utils.data.Subset(full_test, val_indices)
    

    print(f"Train set size: {len(trainset)}, Validation set size: {len(valset)}, Test set size: {len(testset)}")

    return trainset, valset, testset, class_names
    

