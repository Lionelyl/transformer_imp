import copy
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Transformer

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.linear_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.linear_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.linear_v = nn.Linear(embed_dim, embed_dim, bias=False)

        self.scale = self.head_dim ** -0.5

        self.softmax = nn.Softmax(dim=3)
        self.attention_dropout = nn.Dropout(dropout_rate)

        self.linear_out = nn.Linear(embed_dim, embed_dim, bias=False)


    def forward(self, q, k, v, mask=None):
        """
        :param q: [batch_size, L, d_model]
        :param k: [batch_size, S, d_model]
        :param v: [batch_size, S, d_model]

        :param mask: sequence mask [1, L, L]
        """

        d_k = self.head_dim
        d_v = self.head_dim
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)

        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1,2)                    # [batch_size, num_heads, len_q, d_k]
        k = k.transpose(1,2).transpose(2,3)     # [batch_size, num_heads, d_k, len_k]
        v = v.transpose(1,2)                    # [batch_size, num_heads, len_k, d_v]

        x = torch.matmul(q, k)                  # [batch_size, num_heads, len_q, len_k]
        x.mul_(self.scale)

        # todo: mask, only decoder first self-attention layer have tgt_mask
        if mask is not None:
            x.masked_fill_(mask.unsqueeze(1) == 1, -1e9) # [1, 1, len_q, len_k]

        x = self.softmax(x)
        x = torch.matmul(x, v)                  # [batch_size, num_heads, len_q, d_v]

        x = x.transpose(1,2).contiguous()
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.linear_out(x)
        return x

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model=512, dim_feedforward=2048, dropout_rate=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, dim_feedforward=2048, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()

        # multi-head attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.self_attention_dropout = nn.Dropout(dropout_rate)
        self.self_attention_norm = nn.LayerNorm(d_model,  eps=1e-6)

        # feed forward network
        self.feed_forward_network = FeedForwardNetwork(d_model, dim_feedforward, dropout_rate)
        self.feed_forward_dropout = nn.Dropout(dropout_rate)
        self.feed_forward_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, mask):

        # multi-head attention
        y = self.self_attention(x, x, x, mask)
        y = self.self_attention_dropout(y)
        y = y + x
        x = self.self_attention_norm(y)

        # feed forward network
        y = self.feed_forward_network(x)
        y = self.feed_forward_dropout(y)
        y = y + x
        x = self.feed_forward_norm(y)

        return x

class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers=6):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.layers = clone_modules(encoder_layer, num_layers)

    def forward(self, x, mask):
        output = x

        for module in self.layers:
            output = module(output, mask)

        return output

class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, dim_feedforward=2048, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.self_attention_dropout = nn.Dropout(dropout_rate)
        self.self_attention_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.enc_dec_attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.enc_dec_attention_dropout = nn.Dropout(dropout_rate)
        self.enc_dec_attention_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.feed_forward_network = FeedForwardNetwork(d_model, dim_feedforward, dropout_rate)
        self.feed_forward_dropout = nn.Dropout(dropout_rate)
        self.feed_forward_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, tgt, memory, tgt_mask, memory_mask):

        x = tgt

        # self attention
        y = self.self_attention(x, x, x, tgt_mask)
        y = self.self_attention_dropout(y)
        y = y + x
        x = self.self_attention_norm(y)

        # encoder-decoder attention
        y = self.enc_dec_attention(x, memory, memory, memory_mask)
        y = self.enc_dec_attention_dropout(y)
        y = y + x
        x = self.enc_dec_attention_norm(y)

        # feed forward
        y = self.feed_forward_network(x)
        y = self.feed_forward_dropout(y)
        y = y + x
        x = self.feed_forward_norm(y)

        return x

class Decoder(nn.Module):
    def __init__(self, decoder_layer, num_layers=6):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.layers = clone_modules(decoder_layer, num_layers)

    def forward(self, tgt, memory, tgt_mask, memory_mask):

        output = tgt

        for module in self.layers:
            output = module(output, memory, tgt_mask, memory_mask)

        return output

class Transformer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048,dropout_rate=0.1):
        super(Transformer, self).__init__()

        encoder_layer = EncoderLayer(d_model, num_heads, dim_feedforward, dropout_rate)
        self.encoder = Encoder(encoder_layer, num_encoder_layers)

        decoder_layer = DecoderLayer(d_model, num_heads, dim_feedforward, dropout_rate)
        self.decoder = Decoder(decoder_layer, num_decoder_layers)

        self.set_parameters()

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):

        memory = self.encoder(src, src_mask)
        output = self.decoder(tgt, memory, tgt_mask, memory_mask)

        return output

    def set_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Embeddings, self).__init__()

        self.lut = nn.Embedding(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * torch.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    # "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


def clone_modules(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def get_padding_mask(target, pad):
    mask = (target == pad).unsqueeze(-2)  # unsqueeze(-2) ?
    return mask

def get_sequence_mask(target_len):

    ones = torch.ones(target_len, target_len, dtype=torch.uint8)

    mask = torch.triu(ones, diagonal=1).unsqueeze(0)   # [1, tgt_len, tgt_len]

    return mask






























