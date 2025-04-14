import torch
import torch.nn as nn
import torchvision

# Funzione per ottenere il modello selezionato.
def get_model(model_name, num_classes=196):
    
    if model_name == "resnet18":
        model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        
        # Funzione per congelare i layer
        def set_parameter_requires_grad(net, req_grad=False):
            for name, param in net.named_parameters():
                if "layer3" not in name and "layer4" not in name:
                    param.requires_grad = req_grad
                #if "layer4" not in name:
                    #param.requires_grad = req_grad
                    
        # Congela tutti i layer tranne layer3 e layer4
        set_parameter_requires_grad(model, req_grad=False)

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == "resnet34":
        model = torchvision.models.resnet34(weights='IMAGENET1K_V1')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    else:
        raise ValueError(f"Model {model_name} is not supported.")
    
    return model