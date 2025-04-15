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
from neural_collapse import NeuralCollapse
import pandas as pd
import wandb
from collections import Counter
from imagenet_experiments_prep import RANDOM_TASKS, FAVORITE_LABEL_MAPPINGS, RANDOM_CLASS_REMAPPINGS
 



def main():

    
    # Set up the paths
    print("Setting up paths")
    root = "./data"
    nc_dir = "./nc_new_2"
    torch.set_num_threads(4)
    use_wandb = False
    n_classes_per_labels = np.arange(1, 11)
    # normalize = True
    apply_label_mapping = False
    # apply_mix_and_match = "FashionMNIST"
    if not os.path.exists(nc_dir):
        os.makedirs(nc_dir)
    # for n_classes_per_label in n_classes_per_labels:
    parameters = {
        "dataset_name": ["ImageNet"],  #["MNIST", "FashionMNIST", "CIFAR10"], ##
        "backbone_name": ["resnet50", "swin_b"],#["resnet18"],#["resnet18", "vit_b_16",
        "normalize": [True],#, False], 
        # "label_mapping": [FAVORITE_LABEL_MAPPINGS[i] for i in range(1,6)],
        "label_mapping": [{}], # RANDOM_CLASS_REMAPPINGS[n_classes_per_label],        
        "apply_mix_and_match": [None]#["FashionMNIST", "CIFAR10", "MNIST", None]
    }
    
    combinations = list(ParameterGrid(parameters))
    
    # Use a list comprehension to filter combinations
    filtered_combinations = [
        combination for combination in combinations
        if not (
            combination["dataset_name"] == combination["apply_mix_and_match"]
            or (combination["label_mapping"] and combination["apply_mix_and_match"] is not None)
        )
    ]

    # Update the original list if needed
    combinations = filtered_combinations # 102 combinations
    
    results = {"dataset":[], "backbone":[], "normalize":[], "label_mapping":[], "apply_mix_and_match":[], "nc_1":[], "nc_2":[], "en_ammar":[]}
    for idx, combination in enumerate(combinations):
        print(f"Running combination {idx+1}/{len(combinations)}")
        if use_wandb:
            wandb.init(
                project="neural collapse",
                name= f"run_{idx}"
            )
            wandb.config.update(combination)
        dataset_name = combination["dataset_name"]
        backbone = combination["backbone_name"]
        normalize = combination["normalize"]
        label_mapping = combination["label_mapping"] # TODO
        apply_mix_and_match = combination["apply_mix_and_match"]
        divide = "all" #average" #"random subset" # ,  "all"
        # dataset_name = "ImageNet"
        # backbone = "resnet18"
        # normalize = True
        # label_mapping = FAVORITE_LABEL_MAPPINGS[2]
        # apply_mix_and_match = None #"FashionMNIST"
        train = True if not dataset_name == "ImageNet" else False
        embeddings_path, embeddings_filename = get_embeddings_path(root, dataset_name, backbone, train=train)
        embeddings_dataset = EmbeddingDataset(os.path.join(embeddings_path, embeddings_filename), label_mapping=label_mapping)
        embeddings = embeddings_dataset[:][0]
        labels = embeddings_dataset[:][1]
        wanted_labels = ~torch.isin(labels, torch.tensor([-1]))
        embeddings = embeddings[wanted_labels]
        labels = labels[wanted_labels]

        if apply_mix_and_match is not None:
            embeddings_dataset = mix_and_match(dataset_name, apply_mix_and_match, root, backbone)
            embeddings = embeddings_dataset[:][0]
            labels = embeddings_dataset[:][1]    
        if normalize:
            embeddings = embeddings / (torch.norm(embeddings, dim=1, keepdim=True)+torch.finfo(embeddings.dtype).eps)
        

        
        
        
        if divide == "random subset":
            code = "rs_"
            intra_all = torch.zeros((len(RANDOM_TASKS), 10, embeddings.size(1), embeddings.size(1)))
            inter_all = torch.zeros((len(RANDOM_TASKS), embeddings.size(1), embeddings.size(1)))
            nc_1_all = torch.zeros(len(RANDOM_TASKS))
            en_ammar_all = torch.zeros(len(RANDOM_TASKS))
            nc_2_all = torch.zeros((len(RANDOM_TASKS), embeddings.size(1)))
            for idx, i in enumerate(RANDOM_TASKS):
                random_labels = torch.isin(labels, torch.tensor(i))
                embeddings_sub = embeddings[random_labels]
                labels_sub = labels[random_labels]
                nc = NeuralCollapse(embeddings_sub, labels_sub)
                intra_all[idx], inter_all[idx] = nc.covariance()
                nc_1_all[idx] = nc.nc_1(intra_all[idx], inter_all[idx])
                en_ammar_all[idx] = nc.en_ammar()
                nc_2_all[idx] = nc.nc_2()
            inter = inter_all.mean(dim=0)
            intra = intra_all.mean(dim=0)
            nc_1 = nc_1_all.mean()
            en_ammar = en_ammar_all.mean()
            nc_2 = nc_2_all.mean(dim=0)
            
        
        elif divide == "average":
            code = "av_"
            intra_all = torch.zeros((100, 10, embeddings.size(1), embeddings.size(1)))
            inter_all = torch.zeros((100, embeddings.size(1), embeddings.size(1)))
            nc_1_all = torch.zeros(100)
            en_ammar_all = torch.zeros(100)
            nc_2_all = torch.zeros((100, embeddings.size(1)))
            for i in range(100):
                random_labels = torch.isin(labels, torch.tensor(range(i*10, (i+1)*10)))
                embeddings_sub = embeddings[random_labels]
                labels_sub = labels[random_labels]
                nc = NeuralCollapse(embeddings_sub, labels_sub)
                intra_all[i], inter_all[i] = nc.covariance()
                nc_1_all[i] = nc.nc_1(intra_all[i], inter_all[i])
                en_ammar_all[i] = nc.en_ammar()
                nc_2_all[i] = nc.nc_2()
            inter = inter_all.mean(dim=0)
            intra = intra_all.mean(dim=0)
            nc_1 = nc_1_all.mean()
            en_ammar = en_ammar_all.mean()
            nc_2 = nc_2_all.mean(dim=0)
        
        
        elif divide == "all":
            code = ""
            nc = NeuralCollapse(embeddings, labels)
            intra, inter = nc.covariance()
            nc_1 = nc.nc_1(intra, inter)
            # en_ammar = nc.en_ammar()
            # nc_2 = nc.nc_2()
            
            
        
        # results["dataset"].append(dataset_name)
        # results["backbone"].append(backbone)
        # results["normalize"].append(normalize)
        # results["label_mapping"].append(Counter(label_mapping.values())[0])
        # results["apply_mix_and_match"].append(apply_mix_and_match)
        # results["nc_1"].append(nc_1)
        # results["nc_2"].append(torch.norm(nc_2).item())
        # results["en_ammar"].append(en_ammar.item())
            
                
        # results = pd.DataFrame(results)
        # results.to_csv(os.path.join(nc_dir, f"random_class_remapping_imagenet_{n_classes_per_label}.csv"), index=False)
        
            
            
            
        torch.save({"intra": intra, "inter": inter}, os.path.join(nc_dir, f"{code}{dataset_name}_{backbone}_norm_{normalize}_lm_{apply_label_mapping}_m&m_{apply_mix_and_match}_cov.pth"))
        torch.save({"nc_1":nc_1}, os.path.join(nc_dir, f"{code}{dataset_name}_{backbone}_norm_{normalize}_lm_{apply_label_mapping}_m&m_{apply_mix_and_match}_nc_1.pth"))
        # torch.save(nc_2, os.path.join(nc_dir, f"{code}{dataset_name}_{backbone}_norm_{normalize}_lm_{apply_label_mapping}_m&m_{apply_mix_and_match}_nc_2.pth"))
        # torch.save(en_ammar.item(), os.path.join(nc_dir, f"{code}{dataset_name}_{backbone}_norm_{normalize}_lm_{apply_label_mapping}_m&m_{apply_mix_and_match}_en_ammar.pth"))
        # torch.save(nc_2, os.path.join(nc_dir, f"div_1_{dataset_name}_{backbone}_norm_{normalize}_lm_{apply_label_mapping}_m&m_{apply_mix_and_match}_nc_2.pth"))
        if use_wandb:
            wandb.log({"intra_class_covariance": intra, "inter_class_covariance": inter,"nc_1_zhu": nc_1})
            wandb.finish()
            



# def mix_and_match(dataset_1, dataset_2, root, backbone_name):
#     embeddings_path_1, embeddings_filename_1 = get_embeddings_path(root, dataset_1, backbone_name, train=False if dataset_1 == "ImageNet" else True)
#     embeddings_dataset_1 = EmbeddingDataset(os.path.join(embeddings_path_1, embeddings_filename_1))
#     embeddings_path_2, embeddings_filename_2 = get_embeddings_path(root, dataset_2, backbone_name, train=False if dataset_2 == "ImageNet" else True)
#     embeddings_dataset_2 = EmbeddingDataset(os.path.join(embeddings_path_2, embeddings_filename_2))
#     embeddings_1 = embeddings_dataset_1[:][0]
#     embeddings_2 = embeddings_dataset_2[:][0]
#     labels_1 = embeddings_dataset_1[:][1]
#     labels_2 = embeddings_dataset_2[:][1]
#     unique_labels_1 = np.unique(labels_1)
#     unique_labels_2 = np.unique(labels_2)
#     unique_labels = unique_labels_1 if len(unique_labels_1) < len(unique_labels_2) else unique_labels_2
#     mixed_embeddings = torch.zeros((0, embeddings_1.size(1)))
#     labels_new = []
#     for label in unique_labels:
#         unique_1 = embeddings_1[labels_1 == label]
#         unique_2 = embeddings_2[labels_2 == label]
#         mixed_embeddings = torch.vstack([mixed_embeddings, unique_1, unique_2])
#         labels_new += [label]*(len(unique_1) + len(unique_2))
#     return (mixed_embeddings, torch.tensor(labels_new))


if __name__ == "__main__":
    main()

