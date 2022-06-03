
import torch
import torch.nn as nn
from model.utils import pad_mask, masked_attn_mask
from model.sub_layers import Embedding_Layer, Encoder, Decoder
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
class Transformer(nn.Module):
    def __init__(self, num_token, max_seq_len, dim_model, d_k=64, d_v=64, n_head=8, dim_hidden=2048, d_prob=0.1, n_enc_layer=6, n_dec_layer=6):
        super(Transformer, self).__init__()

        """
        each variable is one of example, so you can change it, it's up to your coding style.
        """
        self.num_token = num_token
        self.max_seq_len = max_seq_len
        self.embed = Embedding_Layer(num_token=num_token, dim_model=dim_model, max_seq_len=max_seq_len, d_prob=d_prob)
        self.encoder = Encoder(dim_model, d_k, d_v, n_head, dim_hidden, d_prob, n_enc_layer)
        self.decoder = Decoder(dim_model, d_k, d_v, n_head, dim_hidden, d_prob, n_dec_layer)
        self.linear = nn.Linear(dim_model, num_token)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, src, tgt, teacher_forcing=True):

        """
        ToDo: feed the input to Encoder and Decoder
        """

        return

    def init_weights(self):
        for n, p in self.named_parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)