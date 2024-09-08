import torch
import torchvision
import torch.nn as nn
import numpy as np


def select_device():
    if torch.cuda.is_available(): print(torch.cuda.get_device_name()) 
    else: print('cpu')
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return default_device

class FeatureExtractor(nn.Module):

    def __init__(self, weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1):
        super(FeatureExtractor, self).__init__()
        Alex = torchvision.models.alexnet(weights=weights)

        Alex.classifier[6] = torch.nn.Linear(Alex.classifier[6].in_features, 2)
        # Load the saved weights from the file
        Alex.load_state_dict(torch.load('alex_weights.pth'))
        
        self.feature = Alex.features
        self.classifier = Alex.classifier[:6] 

    def forward(self, x):
        x = self.feature(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def extract(self, x):
        with torch.no_grad():
            x = self.forward(x)
            x = x.cpu().detach().numpy()
            x = (x / (np.linalg.norm(x,axis=1))[:, np.newaxis])
        return x

def distance(query_fv, dataset_fv):
    dist = []
    for fv in dataset_fv:
        dist.append(np.linalg.norm(query_fv-fv))
        indx = np.argmin(np.array(dist))
    return indx, dist[indx]





