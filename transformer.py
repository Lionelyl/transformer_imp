import torch
import torch.nn as nn
from torch.nn import Transformer

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.linear_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.linear_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.linear_v = nn.Linear(embed_dim, embed_dim, bias=False)

        self.scale = self.head_dim ** -0.5

        self.softmax = nn.Softmax(dim=3)
        self.attention_dropout = nn.Dropout(dropout_rate)

        self.linear_out = nn.Linear(embed_dim, embed_dim, bias=False)


    def forward(self, q, k, v, mask=None, cache=None):
        """
        :param q: [batch_size, L, d_model]
        :param k: [batch_size, S, d_model]
        :param v: [batch_size, S, d_model]
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

        # todo: mask
        if mask is not None:
            x.masked_fill_(mask.unsqueeze(1), -1e9)

        x = self.softmax(x)
        x = torch.matmul(x, v)                  # [batch_size, num_heads, len_q, d_v]

        x = x.transpose(1,2).contiguous()
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.linear_out(x)
        return x

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model=512, dim_feedforward=2048):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
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
        self.feed_forward_network = FeedForwardNetwork(d_model, dim_feedforward)
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
    def __init__(self, d_model=512, num_heads=8, num_layers=6, dim_feedforward=2048, dropout_rate=0.1):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        encoder_list = [EncoderLayer(d_model, num_heads, dim_feedforward, dropout_rate)
                        for _ in range(num_layers)]
        self.layers = nn.ModuleList(encoder_list)

    def forward(self, x, mask):
        output = x

        for model in self.layers:
            output = model(output, mask)

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

        self.feed_forward_network = FeedForwardNetwork(d_model, dim_feedforward)
        self.feed_forward_dropout = nn.Dropout(dropout_rate)
        self.feed_forward_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, tgt, memory, tgt_mask, memory_mask, cache):

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
    def __init__(self, d_model=512, num_heads=8, num_layers=6, dim_feedforward=2048, dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        decoer_list = [DecoderLayer(d_model, num_heads, dim_feedforward, dropout_rate)
                       for _ in range(num_layers)]
        self.layers = nn.ModuleList(decoer_list)

    def forward(self, tgt, memory, tgt_mask, memory_mask, cache):

        output = tgt

        for model in self.layers:
            output = model(output, memory, tgt_mask, memory_mask, cache)

        return output

class Transformer(nn.Module):
    def __init__(self, input_vocab_szie, target_vocab_size,
                 num_layers=6,
                 d_model = 512,
                 dim_feedforward=2048,
                 dropout_rate=0.1,
                 share_target_embedding=True):
        super(Transformer, self).__init__()


    def forward(self, input, target):
        return None


































