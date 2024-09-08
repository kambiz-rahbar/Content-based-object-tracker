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

    def __init__(self,  weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1):
        super(FeatureExtractor, self).__init__()
        VGG = torchvision.models.vgg16(weights=weights)
        self.feature = VGG.features
        self.classifier = VGG.classifier[:5] 

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





