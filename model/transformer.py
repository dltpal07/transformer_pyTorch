import torch
import torch.nn as nn
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

        positional_encoded_src = self.embed(src)
        src_pad_mask = self.make_mask(src)
        enc_output = self.encoder(positional_encoded_src, src_pad_mask.bool())

        if teacher_forcing:
            positional_encoded_tgt = self.embed(tgt)
            tgt_pad_mask = self.make_mask(tgt)
            tgt_src_pad_mask = torch.bmm(tgt_pad_mask, src_pad_mask.transpose(-1, -2))
            dec_output = self.decoder(positional_encoded_tgt, enc_output, tgt_pad_mask.bool(), tgt_src_pad_mask.bool())
            outputs = self.softmax(self.linear(dec_output))

        else:
            outputs = []
            cur_tgt = tgt
            for i in range(self.max_seq_len):
                cur_tgt_embed = self.embed(cur_tgt, i)
                dec_output = self.decoder(cur_tgt_embed, enc_output)
                dec_output = self.softmax(self.linear(dec_output))
                outputs.append(dec_output[:, i].unsqueeze(1))
                cur_tgt[:,i+1] = torch.argmax(dec_output, dim=-1)[:,i]
            outputs = torch.cat(outputs, dim=1)

        return outputs

    def init_weights(self):
        for n, p in self.named_parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)

    def make_mask(self, x):
        pad_mask = torch.where(x==2, 0., 1.)
        pad_mask = pad_mask.unsqueeze(-1)
        pad_mask = torch.bmm(pad_mask, pad_mask.transpose(-1, -2))
        return pad_mask