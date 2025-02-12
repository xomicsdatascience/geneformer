# geneformer
A recreation of the [Geneformer model described by Theodoris et al.](https://doi.org/10.1038/s41586-023-06139-9) using [AttentionSmithy](https://github.com/xomicsdatascience/AttentionSmithy).

# Main Files
## scripts/1_train_model.py
This file is the pre-training script for the geneformer foundation model.

## scripts/2_fine_tune_model.py
This file is a training script for fine-tuning a pretrained geneformer model for a cell classification task.

## src/geneformer/Geneformer.py
The code for the (pre-trained) geneformer foundation model. It was written using pytorch lightning for readability, and thus outlines the construction of the model, the forward pass process, and how that looks for training and validation steps.

## src/geneformer/data/GeneformerDataModule.py
The code for preparing the data module used in training and validating the geneformer foundation model. It is made to be used with the pytorch lightning Trainer class, as called in model training scripts.

## src/geneformer/fine_tuned_model/GeneformerForSequenceClassification.py
The code for the fine-tuned geneformer model. It was also written using pytorch lightning.

## src/geneformer/fine_tuned_model/GeneformerDataModuleForSequenceClassification.py
The code for preparing the data module used in training and validating the fine-tuned geneformer model.

# Citations, links
Theodoris, C.V., Xiao, L., Chopra, A. et al. Transfer learning enables predictions in network biology. Nature 618, 616–624 (2023). https://doi.org/10.1038/s41586-023-06139-9

https://huggingface.co/ctheodoris/Geneformer
