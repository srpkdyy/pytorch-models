import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_pos=512):
        super().__init__()

        pos = torch.arange(max_pos)

        freq = torch.arange(dim//2) / dim
        freq = (freq * torch.tensor(10000).log()).exp()

        x = rearrange(pos, 'L -> L 1') / freq
        x = rearrange(x, 'L D -> L D 1')

        pe = torch.cat((x.sin(), x.cos()), dim=-1)
        self.pe = rearrange(pe, 'L D sc -> L (D sc)')

    def forward(self, n):
        return self.pe[:n]


class MultiHeadAttention(nn.Module):
    def __init__(self, x):
        pass

    def forward(self, query, key_value=None):
        pass


class EncoderBlock(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


class Encoder(nn.Module):
    def __init__(self, n_layers=6, dim=512):
        super().__init__()

        self.pos_bias = PosEnc()


class Decoder(nn.Module):
    def __init__(self, n_layers=6, dim=512):
        super().__init__()

        self.dim = dim

        self.pos_bias = PosEnc()



class Transformer(nn.Module):
    def __init__(self, encoder_config, decoder_config):
        super().__init__()

        self.encoder = Encoder(**encoder_config)
        self.decoder = Decoder(**decoder_config)
        self.linear = nn.Linear(self.decoder.dim, 1)


