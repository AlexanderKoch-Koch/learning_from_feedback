import torch


def generate_mask(size: int, device):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones((size, size), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def pad(tensor, desired_size):
    # returns tensor with zero padding in last dimension so that shape[-1] == desired_size
    padding_shape = list(tensor.shape)
    padding_shape[-1] = desired_size - tensor.shape[-1]
    return torch.cat((tensor, torch.zeros(padding_shape, device=tensor.device)), dim=-1)

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=300):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        # pe = torch.zeros(max_len, d_model)
        # position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        # self.register_buffer('pe', pe)
        self.pe = torch.nn.Parameter(torch.randn(max_len, 1, self.d_model))

    def forward(self, x):
        # x = x * math.sqrt(self.d_model)
        # x = x + self.pe[:x.size(0), :]
        x = x + self.pe[:x.shape[0]]
        return x
