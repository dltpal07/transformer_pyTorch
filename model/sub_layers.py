import torch
import torch.nn as nn
import math


class Embedding_Layer(nn.Module):
    def __init__(self, num_token, dim_model, max_seq_len, d_prob):
        super(Embedding_Layer, self).__init__()
        self.num_token = num_token
        self.dim_model = dim_model
        self.max_seq_len = max_seq_len
        self.d_prob = d_prob
        self.emb = nn.Embedding(num_token, dim_model)

        pos_enc = torch.zeros((self.max_seq_len, self.dim_model))
        for pos in range(self.max_seq_len):
            for idx in range(0, self.dim_model, 2):
                pos_enc[pos, idx] = torch.sin(torch.tensor(pos / (10000.0) ** (float(idx) / self.dim_model)))
                pos_enc[pos, idx + 1] = torch.cos(torch.tensor(pos / (10000.0) ** (float(idx) / self.dim_model)))

        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        x = self.emb(x) * math.sqrt(self.dim_model)
        x += self.pos_enc
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, d_prob):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.softmax = nn.Softmax(dim=-1)
        self.drop_out = nn.Dropout(d_prob)

    def forward(self, x_q, x_k, x_v, pad_mask):
        dot_product = torch.matmul(x_q, x_k.transpose(-1, -2))
        scaled_dot_product = dot_product / math.sqrt(self.d_k)
        if pad_mask != None:
            scaled_dot_product = scaled_dot_product.masked_fill(pad_mask == False, -1e9)
        reg_scaled_dot_product = self.softmax(scaled_dot_product)
        reg_scaled_dot_product = self.drop_out(reg_scaled_dot_product)
        scaled_dot_product_attn = torch.matmul(reg_scaled_dot_product, x_v)
        return scaled_dot_product_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_model, d_k, d_v, n_head, d_prob):
        super(MultiHeadAttention, self).__init__()

        self.dim_model = dim_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_head = n_head

        self.w_q = nn.Linear(dim_model, n_head * d_k)
        self.w_k = nn.Linear(dim_model, n_head * d_k)
        self.w_v = nn.Linear(dim_model, n_head * d_v)
        self.w_o = nn.Linear(n_head * d_v, dim_model)

        self.scaled_dot_prod = ScaledDotProductAttention(d_k, d_prob)

    def forward(self, q, k, v, pad_mask):
        x_q = self.w_q(q).view(len(q), -1, self.n_head, self.d_k).transpose(1, 2)
        x_k = self.w_k(k).view(len(k), -1, self.n_head, self.d_k).transpose(1, 2)
        x_v = self.w_v(v).view(len(v), -1, self.n_head, self.d_v).transpose(1, 2)
        if pad_mask != None:
            pad_mask = pad_mask.unsqueeze(1)
            pad_mask = pad_mask.expand(-1, self.n_head, -1, -1)
        scaled_dot_product_attn = self.scaled_dot_prod(x_q, x_k, x_v, pad_mask)
        scaled_dot_product_attn = scaled_dot_product_attn.transpose(1, 2)
        scaled_dot_product_attn = scaled_dot_product_attn.reshape(len(v), -1, self.d_v * self.n_head)
        output = self.w_o(scaled_dot_product_attn)
        return output, q


class MaskedScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, d_prob):
        super(MaskedScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.softmax = nn.Softmax(dim=-1)
        self.drop_out = nn.Dropout(d_prob)

    def forward(self, x_q, x_k, x_v, pad_mask):
        dot_product = torch.matmul(x_q, x_k.transpose(-1, -2))
        scaled_dot_product = dot_product / math.sqrt(self.d_k)
        true_arr = torch.ones_like(scaled_dot_product)
        if pad_mask != None:
            mask = torch.tril(true_arr).bool()
            scaled_dot_product = scaled_dot_product.masked_fill(mask==False, -1e9)
            scaled_dot_product = scaled_dot_product.masked_fill(pad_mask==False, -1e9)
        reg_scaled_dot_product = self.softmax(scaled_dot_product)
        #print(reg_scaled_dot_product[0])
        reg_scaled_dot_product = self.drop_out(reg_scaled_dot_product)
        scaled_dot_product_attn = torch.matmul(reg_scaled_dot_product, x_v)
        return scaled_dot_product_attn


class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, dim_model, d_k, d_v, n_head, d_prob):
        super(MaskedMultiHeadAttention, self).__init__()
        self.dim_model = dim_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_head = n_head

        self.w_q = nn.Linear(dim_model, n_head * d_k)
        self.w_k = nn.Linear(dim_model, n_head * d_k)
        self.w_v = nn.Linear(dim_model, n_head * d_v)
        self.w_o = nn.Linear(n_head * d_v, dim_model)

        self.masked_scaled_dot_prod = MaskedScaledDotProductAttention(d_k, d_prob)

    def forward(self, q, k, v, pad_mask):
        x_q = self.w_q(q).view(len(q), -1, self.n_head, self.d_k).transpose(1, 2)
        x_k = self.w_k(k).view(len(k), -1, self.n_head, self.d_k).transpose(1, 2)
        x_v = self.w_v(v).view(len(v), -1, self.n_head, self.d_v).transpose(1, 2)
        if pad_mask != None:
            pad_mask = pad_mask.unsqueeze(1)
            pad_mask = pad_mask.expand(-1, self.n_head, -1, -1)
        scaled_dot_product_attn = self.masked_scaled_dot_prod(x_q, x_k, x_v, pad_mask)
        scaled_dot_product_attn = scaled_dot_product_attn.transpose(1, 2)
        scaled_dot_product_attn = scaled_dot_product_attn.reshape(len(v), -1, self.d_v * self.n_head)
        output = self.w_o(scaled_dot_product_attn)
        return output, v


class FFNN(nn.Module):
    def __init__(self, dim_model, dim_hidden, d_prob):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(dim_model, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_model)

        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(d_prob)

    def forward(self, x):
        output = self.fc2(self.drop_out(self.relu(self.fc1(x))))
        return output, x


class EncoderLayer(nn.Module):
    def __init__(self, dim_model, d_k, d_v, n_head, dim_hidden, d_prob):
        super(EncoderLayer, self).__init__()
        self.dim_model = dim_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_head = n_head
        self.dim_hidden = dim_hidden
        self.d_prob = d_prob

        self.multi_head_attn = MultiHeadAttention(dim_model, d_k, d_v, n_head, d_prob)
        self.ffnn = FFNN(dim_model, dim_hidden, d_prob)

        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.layer_norm2 = nn.LayerNorm(dim_model)

        self.drop_out = nn.Dropout(d_prob)

    def forward(self, x, src_pad_mask):
        x, x_residual = self.multi_head_attn(x, x, x, src_pad_mask) # Q = K = V
        x = self.layer_norm1(x + x_residual)
        x, x_residual = self.ffnn(x)
        x = self.layer_norm2(x + x_residual)
        return x


class Encoder(nn.Module):
    def __init__(self, dim_model, d_k, d_v, n_head, dim_hidden, d_prob, n_enc_layer):
        super(Encoder, self).__init__()
        self.dim_model = dim_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_head = n_head
        self.dim_hidden = dim_hidden
        self.d_prob = d_prob
        self.n_enc_layer = n_enc_layer

        self.enc_layers = nn.ModuleList([EncoderLayer(dim_model, d_k, d_v, n_head, dim_hidden, d_prob) for _ in range(n_enc_layer)])

    def forward(self, x, src_pad_mask):
        for layer in self.enc_layers:
            x = layer(x, src_pad_mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, dim_model, d_k, d_v, n_head, dim_hidden, d_prob):
        super(DecoderLayer, self).__init__()
        self.dim_model = dim_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_head = n_head
        self.dim_hidden = dim_hidden
        self.d_prob = d_prob

        self.masked_multi_head_attention = MaskedMultiHeadAttention(dim_model, d_k, d_v, n_head, d_prob)
        self.multi_head_attention = MultiHeadAttention(dim_model, d_k, d_v, n_head, d_prob)
        self.ffnn = FFNN(dim_model, dim_hidden, d_prob)

        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.layer_norm2 = nn.LayerNorm(dim_model)
        self.layer_norm3 = nn.LayerNorm(dim_model)

        self.drop_out = nn.Dropout(d_prob)
    def forward(self, x, enc_output, tgt_pad_mask, src_tgt_pad_mask):
        x, x_residual = self.masked_multi_head_attention(x, x, x, tgt_pad_mask)  # Q = K = V
        x = self.layer_norm1(x + x_residual)
        x, x_residual = self.multi_head_attention(x, enc_output, enc_output, src_tgt_pad_mask)
        x = self.layer_norm2(x + x_residual)
        x, x_residual = self.ffnn(x)
        x = self.layer_norm3(x + x_residual)

        return x


class Decoder(nn.Module):
    def __init__(self, dim_model, d_k, d_v, n_head, dim_hidden, d_prob, n_dec_layer):
        super(Decoder, self).__init__()
        self.dim_model = dim_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_head = n_head
        self.dim_hidden = dim_hidden
        self.d_prob = d_prob
        self.n_dec_layer = n_dec_layer

        self.dec_layers = nn.ModuleList([DecoderLayer(dim_model, d_k, d_v, n_head, dim_hidden, d_prob) for _ in range(n_dec_layer)])
        # layers = []
        # for i in range(6):
        #     layers.append(DecoderLayer(dim_model, d_k, d_v, n_head, dim_hidden, d_prob))
        # self.dec_layers = nn.ModuleList(layers)

    def forward(self, x, enc_output, tgt_pad_mask=None, src_tgt_pad_mask=None):
        for layer in self.dec_layers:
            x = layer(x, enc_output, tgt_pad_mask, src_tgt_pad_mask)
        return x



