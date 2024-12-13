from torch import nn

class MaskedLoss(nn.Module):

    def __init__(self, hidden, vocab_size, padding_token):
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss(ignore_index=padding_token)

    def forward(self, x, original_masked_value_tensor):
        masked_logits = self.softmax(self.linear(x))
        return self.criterion(masked_logits.transpose(1, 2), original_masked_value_tensor)
