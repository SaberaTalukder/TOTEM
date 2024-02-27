import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer

"""
This files contains helper models that convert codes to time. 

* Transformer: Is a transformer model that takes in as input a sequence of length N 
               of D-dimensional codes (N, B, D) and predicts a N-length sequence in time (N, B, C)
               where C is the compression factor (aka each element in the sequence predict C time steps)

* MuStdModel: Is a simple MLP that takes as input the past time series (B, Tin) 
                and predicts the mean and std of the future time series
"""


def conv_lout(lin, kernel_size, padding=0, stride=1, dilation=1):
    return math.floor(
        (lin + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    )


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.0, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )


class ResMLPBlock(nn.Module):
    def __init__(self, hidden_dim=128, res_dim=128):
        super(ResMLPBlock, self).__init__()

        self.fc1 = nn.Linear(hidden_dim, res_dim)
        self.fc2 = nn.Linear(res_dim, hidden_dim)

    def forward(self, x):
        y = self.fc1(x)
        y = F.relu(y)
        y = self.fc2(y)
        x = F.relu(x + y)
        return x


class ResMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, res_dim, nblocks):
        super(ResMLP, self).__init__()

        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.resblocks = nn.ModuleList(
            ResMLPBlock(hidden_dim=hidden_dim, res_dim=res_dim) for _ in range(nblocks)
        )
        self.fc_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc_in(x))
        for resblock in self.resblocks:
            x = resblock(x)
        x = self.fc_out(x)
        return x


class SimpleMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims, dropout=0.0):
        super(SimpleMLP, self).__init__()

        self.nlayers = len(hidden_dims)

        layers = []
        dim = in_dim
        for i in range(self.nlayers):
            layer = nn.Linear(dim, hidden_dims[i])
            layers.append(layer)
            dim = hidden_dims[i]
        self.fcs = nn.ModuleList(layers)
        self.fc_out = nn.Linear(dim, out_dim)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        for fc in self.fcs:
            x = F.relu(fc(x))
            x = self.dropout(x)
        x = self.fc_out(x)

        return x


class SimpleConv1d(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims):
        super(SimpleConv1d, self).__init__()

        self.conv1 = nn.Conv1d(1, 64, kernel_size=4, stride=2)
        lout = conv_lout(in_dim, kernel_size=4, stride=2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=4, stride=2)
        lout = conv_lout(lout, kernel_size=4, stride=2)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=4, stride=2)
        lout = conv_lout(lout, kernel_size=4, stride=2)

        dim = lout * 256
        fcs = []
        for hidden_dim in hidden_dims:
            fcs.append(nn.Linear(dim, hidden_dim))
            dim = hidden_dim
        self.linears = nn.ModuleList(fcs)
        self.linear_out = nn.Linear(dim, 2)

    def forward(self, x):
        """
        Args:
            x: tensor of shape (B, Tin)
        Returns:
            out: tensor of shape (B, 2)
        """
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(start_dim=1)
        for fc in self.linears:
            x = F.relu(fc(x))
        x = self.linear_out(x)
        return x


class MuStdModel(nn.Module):
    def __init__(self, Tin, Tout, hidden_dims, dropout=0.0, is_mlp=True):
        super(MuStdModel, self).__init__()

        # mean, std
        if is_mlp:
            self.mustd = SimpleMLP(
                in_dim=Tin + 2, out_dim=2, hidden_dims=hidden_dims, dropout=dropout
            )
        else:
            raise NotImplementedError

        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: of shape (batch_size, Tin)
        Output:
            out: of shape (batch_size, 2)
        """
        mean = torch.mean(x, dim=1, keepdim=True)
        std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = torch.cat((x, mean, std), dim=1)
        x = self.mustd(x)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
        seq_len: int = 5000,
        batch_first: bool = False,
        norm_first: bool = False,
    ):
        super(Transformer, self).__init__()
        self.model_type = "Transformer"
        self.d_model = d_model

        self.has_linear_in = d_in != d_model
        if self.has_linear_in:
            self.linear_in = nn.Linear(d_in, d_model)

        self.pos_encoder = PositionalEncoding(d_model, dropout, seq_len)

        encoder_layers = TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=d_hid,
            dropout=dropout,
            batch_first=batch_first,
            norm_first=norm_first,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.linear_out = nn.Linear(d_model, d_out)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, codes: torch.Tensor, codes_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Arguments:
            time: tensor of shape (batch_size, Tin)
            codes: tensor of shape (seq_len, batch_size, d_in)
            src_mask: tensor of shape (seq_len, seq_len)

        Returns:
            time_toutput: tensor of shape (batch_size, 2)
            codes_output: tensor of shape (seq_len, batch_size, dout)
        """
        if self.has_linear_in:
            codes = self.linear_in(codes)
        codes = self.pos_encoder(codes)
        codes_output = self.transformer_encoder(codes, codes_mask)
        codes_output = self.linear_out(codes_output)  # (seq, batch, dout)

        return codes_output


class XcodeYtimeDecoder(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        seq_in_len: int = 5000,
        seq_out_len: int = 5000,
        dropout: float = 0.0,
        batch_first: bool = False,
        norm_first: bool = False,
    ):
        super(XcodeYtimeDecoder, self).__init__()
        self.model_type = "Transformer"
        self.d_model = d_model

        self.has_linear_in = d_in != d_model
        if self.has_linear_in:
            self.linear_in = nn.Linear(d_in, d_model)

        self.pos_encoder = PositionalEncoding(d_model, dropout, seq_in_len)

        encoder_layers = TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=d_hid,
            dropout=dropout,
            batch_first=batch_first,
            norm_first=norm_first,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.linear_out = nn.Linear(d_model * seq_in_len, seq_out_len)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Arguments:
            x: tensor of shape (seq_in_len, batch_size, d_in)
            x_mask: tensor of shape (seq_in_len, seq_in_len)

        Returns:
            y: tensor of shape (batch_size, seq_out_len)
        """
        if self.has_linear_in:
            x = self.linear_in(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, x_mask)  # (seq_in_len, batch, d_model)

        x = torch.permute(x, (1, 0, 2))
        x = x.flatten(start_dim=1)
        x = self.linear_out(x)  # (batch, seq_out_len)

        return x