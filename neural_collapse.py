import os
import h5py
import numpy as np
import argparse
import torch
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import ParameterGrid
from datasets import EmbeddingDataset
from embeddings import EmbeddingExtractor, get_embeddings_path
# from k8s.imprinting_jobs_generator import backbone_name


class NeuralCollapse:
    
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
        self.unique_labels = np.unique(self.labels)
        self.data_mean = self.embeddings.mean(dim=0)
        self.centered_class_mean = torch.zeros(len(self.unique_labels), self.embeddings.size(1))
        self.centered_class_mean_norm = torch.zeros(len(self.unique_labels), self.embeddings.size(1))
        for idx, label in enumerate(self.unique_labels):
            class_mean = self.embeddings[self.labels == label].mean(dim=0)
            self.centered_class_mean[idx, :] = class_mean - self.data_mean
            self.centered_class_mean_norm[idx, :] = self.centered_class_mean[idx, :] / torch.norm(self.centered_class_mean[idx, :], keepdim=True) 
            
    
    def covariance(self):
        intra_class_covariance = torch.zeros((len(self.unique_labels), self.embeddings.size(1), self.embeddings.size(1)))
        for idx, label in enumerate(self.unique_labels):
            centered_class_data = self.embeddings[self.labels == label] - self.centered_class_mean[idx, :] 
            intra_class_covariance[idx] = (centered_class_data.T @ centered_class_data)/len(centered_class_data)
        inter_class_covariance = (self.centered_class_mean.T @ self.centered_class_mean)/len(self.unique_labels)
        return intra_class_covariance, inter_class_covariance
    
    
    def nc_1(self, intra_m, inter_m):
        nc_1 = torch.linalg.pinv(inter_m) @ intra_m.mean(dim=0) 
        nc_1 = torch.trace(nc_1).item() / len(self.unique_labels) 
        return nc_1


    def nc_2(self):
        nc_2 = torch.zeros((0, self.embeddings.size(1)))
        for ind_1, label in enumerate(self.unique_labels):
            print("labels_subset", label)
            for ind_2, label2 in enumerate(self.unique_labels):
                if label==label2:
                    continue
                nc_2 = torch.vstack([nc_2, torch.abs(self.centered_class_mean_norm[ind_1] - self.centered_class_mean_norm[ind_2])])
        print("number of samples", len(nc_2))
        return nc_2.mean(dim=0) 
    
    
    def en_ammar(self):
        std = torch.std(torch.norm(self.centered_class_mean, dim=1))
        avg = torch.mean(torch.norm(self.centered_class_mean, dim=1))
        return std/avg
    

    
    