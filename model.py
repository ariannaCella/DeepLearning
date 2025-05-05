import torch
import torch.nn as nn
import torchvision

# Funzione per ottenere il modello selezionato.
def get_model(model_name, num_classes=196):
    
    if model_name == "resnet18":
        model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        
        '''
        Congelo tutti i layers -> Cong1234

        # funzione per congelare i layers del modello
        def set_parameter_requires_grad(model, req_grad = False):
            for param in model.parameters():
                param.requires_grad = req_grad
        set_parameter_requires_grad(model, req_grad=False)
        '''


        '''
        Congelo primo e secondo layers -> Cong12
        
        # congela tutti i layer tranne layer3 e layer4
        def set_parameter_requires_grad(net, req_grad=False):
            for name, param in net.named_parameters():
                if "layer3" not in name and "layer4" not in name:
                    param.requires_grad = req_grad
        
        set_parameter_requires_grad(model, req_grad=False)
        '''


        # Sblocca tutti i parametri per l'addestramento -> NOcong
        for param in model.parameters():
            param.requires_grad = True
        

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    else:
        raise ValueError(f"Model {model_name} is not supported.")
    
    return model
