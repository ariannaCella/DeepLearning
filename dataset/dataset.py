import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.model_selection import train_test_split

def get_transforms():
    return {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.03),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.3)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),

            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            
        ]),
        'test_val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }


# suddivisione originale
def load_dataset_50_50(dataset_path, seed, augmented):
    data_transforms = get_transforms()
    # load train ds
    full_trainset = torchvision.datasets.StanfordCars(root=dataset_path, split='train',
                                                 download=False, transform=None)
    
    # load test set
    testset = torchvision.datasets.StanfordCars(root=dataset_path, split='test',
                                                download=False, transform=data_transforms['test_val'])
                                                
    class_names=full_trainset.classes
    labels = np.array([label for _, label in full_trainset._samples])

    # train set in 80% training e 20% validation mantenendo le proporzioni delle classi
    train_indices, val_indices = train_test_split(np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=seed)

    if augmented:
      trainset = torch.utils.data.Subset(torchvision.datasets.StanfordCars(root=dataset_path, split='train', download=False, transform=data_transforms['train']), train_indices)
    else:
      #no aug
      trainset = torch.utils.data.Subset(torchvision.datasets.StanfordCars(root=dataset_path, split='train', download=False, transform=data_transforms['test_val']), train_indices) 
      
    valset = torch.utils.data.Subset(torchvision.datasets.StanfordCars(root=dataset_path, split='train', download=False, transform=data_transforms['test_val']), val_indices)
    
    print(f"Augmented: {augmented}")
    print(f"Train set size: {len(trainset)}, Validation set size: {len(valset)}, Test set size: {len(testset)}")

    return trainset, valset, testset, class_names
    


# suddivisione 70:30
def load_dataset(dataset_path, seed, augmented):
    data_transforms = get_transforms()
    
    if augmented:
      # load train ds 
      trainset = torchvision.datasets.StanfordCars(root=dataset_path, split='train', download=False, transform=data_transforms['train'])
    else:
      # no aug
      trainset = torchvision.datasets.StanfordCars(root=dataset_path, split='train', download=False, transform=data_transforms['test_val'])
    
    class_names=trainset.classes
    
    # load test set
    full_test = torchvision.datasets.StanfordCars(root=dataset_path, split='test',
                                                download=False, transform=data_transforms['test_val'])
                                                
    labels = np.array([label for _, label in full_test._samples])
    
    # test set in 60% test e 40% validation mantenendo le proporzioni delle classi, in questo modo partizione come se avessimo sui dati totali training set (train+val) del 70% e test set 30%
    test_indices, val_indices = train_test_split(np.arange(len(labels)), test_size=0.4, stratify=labels, random_state=seed)
    
    testset = torch.utils.data.Subset(full_test, test_indices)
    valset = torch.utils.data.Subset(full_test, val_indices)
    
    print(f"Augmented: {augmented}")
    print(f"Train set size: {len(trainset)}, Validation set size: {len(valset)}, Test set size: {len(testset)}")

    return trainset, valset, testset, class_names
    
