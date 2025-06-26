"""
Embedding generation script.

This script generates embeddings for all chosen backbones and datasets.
It interfaces with the data loaders and embedding extractors to process
datasets and save feature vectors for later use in experiments.

The script can be specialized to focus on certain classes of a dataset,
particularly useful for large datasets like ImageNet.

NOTE:
For generating ImageNet embeddings (train, val), simply put in the download links
for IMAGENET_URL_DEVKIT_T_1_2 (ILSVRC2012_devkit_t12.tar.gz),
IMAGENET_URL_TRAIN_T_1_2 (ILSVRC2012_img_train.tar), and
IMAGENET_URL_VAL_T_ALL (LSVRC2012_img_val.tar) in your .env file, and run this
file with ImageNet selections below. It will download the data and generate
the desired embeddings.

We have to download all training data, even if we are not interested in all
classes, because it's one big .tar file.
But we definitely want to limit ourselves on specific classes, when we generate
the embeddings, because it would be way too much and unnecessary data otherwise.
For that, we define in the file `imagenet/prep.py` the classes we are interested in.
"""

from prettytable import PrettyTable

from src.data.loader import available_datasets
from src.data.embeddings import EmbeddingExtractor
from src.models.backbone import available_backbones
from src.experiments.imagenet.prep import IMAGENET_CLASS_FOCUS


def main():
    """
    Execute the embedding generation process for configured datasets and backbones.

    This function iterates through all specified datasets and backbone models,
    creating embeddings for both training and test/validation sets. Results are
    displayed in a table format showing processing statistics.
    """
    # Set datasets
    datasets_list = available_datasets
    # datasets_list = ["MNIST"]
    # datasets_list = ["CIFAR100"]
    # datasets_list = ["Places365"]
    # datasets_list = ["MNIST-M"]
    # datasets_list = ["USPS"]
    # datasets_list = ["MNIST-M", "USPS", "SVHN"]

    # Set backbones
    backbones_list = available_backbones
    # backbones_list = ["resnet18"]
    # backbones_list = ["convnextv2-femto-1k-224"]

    device_name = "cuda"  #  "cuda"  # "cpu"
    batch_size = 64

    class_focus = {
        "ImageNet": IMAGENET_CLASS_FOCUS,
    }
    # NOTE: We have list of classes to focus on, so that we do not generate
    #  the embeddings for all the 1k classes, but only for the ones we are
    #  interested in.

    results_table = PrettyTable()

    results_table.field_names = [
        "Dataset",
        "Train",
        "# Samples",
        "Raw Image Resolution",
        "Location",
        "Model Name",
        "Embeddings",
        "# Embeddings",
        "Dim.",
        "Duration (s)",
    ]

    for dataset_name in datasets_list:
        raw_data_root = "imagenet-data" if dataset_name == "ImageNet" else "imprinting-reproduce"
        # raw_data_root = "data" # for local testing

        embedding_root = raw_data_root
        for backbone_idx, backbone_name in enumerate(backbones_list):
            for train in [False, True]:  # [True, False]:
                train_str = "train" if train else "test"
                if dataset_name == "ImageNet" and not train:
                    train_str = "val"
                print(f"Running {backbone_name} x {dataset_name} ({train_str})")

                extractor = EmbeddingExtractor(
                    device_name=device_name,
                    backbone_name=backbone_name,
                    dataset_name=dataset_name,
                    class_focus=class_focus.get(dataset_name, None),
                    batch_size=batch_size,
                    raw_data_root=raw_data_root,
                    embedding_root=embedding_root,
                    train=train,
                )

                result = extractor.run()

                divider = (not train) and (backbone_idx == len(backbones_list) - 1)
                results_table.add_row(result, divider=divider)
                results_table.float_format = "6.0"
                results_table.align["Dataset"] = "l"
                results_table.align["Train"] = "l"
                print(results_table)


if __name__ == "__main__":
    main()
