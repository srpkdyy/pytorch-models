import torch.nn as nn


class PosEnc(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.bias = None

    def forward(self):
        return self.bias


class MultiHeadAttention(nn.Module):
    def __init__(self, x):
        pass

    def forward(self, query, key_value=None):
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


