import os
import torch
import torch.nn.functional as F
# from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm


class Centroid():
    def __init__(self, n_classes, samples_data, samples_tar, Ctemp, CdecayFactor=0.9999):
        self.n_classes = n_classes
        self.centroids = torch.ones((n_classes, n_classes)) / n_classes
        self.CdecayFactor = CdecayFactor
        self.Ctemp = Ctemp
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.samples_data = torch.tensor(samples_data).to(self.device)
        self.samples_tar = samples_tar


    def update_batch(self, model, targets, sample_size=8):

        with torch.no_grad():

            idx = torch.randperm(500)[:sample_size].to(self.device)

            img = torch.index_select(self.samples_data, 1, idx).view(-1, 3, 32, 32)

            logits = model(img).detach()
            output = F.softmax(logits.float(), 1)
            for Class in range(self.n_classes):
                self.centroids[Class] = self.CdecayFactor * self.centroids[Class] + \
                                        (1 - self.CdecayFactor) * torch.mean(
                    output[Class * sample_size: (Class + 1) * sample_size], axis=0).detach().cpu()

        self.centroids = self.centroids / (self.centroids.sum(1)[:, None])

    def update_epoch(self, model, data_loader):
        self.centroids = torch.zeros_like(self.centroids)
        model.train()
        device = next(model.parameters()).device
        for image, target in tqdm(data_loader):
            image, target = image.to(device), target.to(device)
            logit = model(image).detach()

            Classes = target.cpu().unique()
            logit = logit.cpu()
            output = F.softmax(logit.float(), 1)

            for Class in Classes:
                self.centroids[Class] += torch.sum(output[target.cpu() == Class], axis=0)

        self.centroids = self.centroids / (self.centroids.sum(1)[:, None])

    def get_centroids(self, target):
        return torch.index_select(self.centroids, 0, target.cpu()).to(target.device)