import torch
from torch import nn
from geneformer.data import GeneformerDataModule

class GeneformerDataModuleForSequenceClassification(GeneformerDataModule):
    def __init__(self, dataset, batch_size, num_batches_per_megabatch, padding_token, masking_token, output_tokenizer, test_val_size=5e-4):
        super().__init__(dataset, batch_size, num_batches_per_megabatch, padding_token, masking_token, test_val_size)
        self.output_tokenizer = output_tokenizer

    def _collate_function(self, batch):
        input_tensors = [torch.tensor(x['input_ids']) for x in batch]
        input_tensor = nn.utils.rnn.pad_sequence(
            input_tensors,
            batch_first=True,
            padding_value=self.padding_token,
        )
        output_classes = torch.tensor([self.output_tokenizer[x['cell_type']] for x in batch])
        return input_tensor, output_classes