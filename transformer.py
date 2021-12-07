import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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

        :param mask: encoder: src_padding_mask [B, 1, L]
                     decoder: tgt_padding_mask & tgt_sequence_mask [B, S, S]
                     enc_dec: memory_mask = src_padding_mask [B, 1, L]
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

        # Mask
        # encoder: x [b, h, l, l] mask [b, 1, 1, l]  ==> [b, h, l, l]
        # enc_dec: x [b, h, s, l] mask [b, 1, 1, l]  ==> [b, h, s, l]
        # decoder: x [b, h, s, s] mask [b, 1, s, s]  ==> [b, h, s, s]
        if mask is not None:
            x.masked_fill_(mask.unsqueeze(1) != 1, -1e9)

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

class EncoderDecoder(nn.Module):
    def __init__(self, d_model=512, num_heads=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048,dropout_rate=0.1):
        super(EncoderDecoder, self).__init__()

        encoder_layer = EncoderLayer(d_model, num_heads, dim_feedforward, dropout_rate)
        self.encoder = Encoder(encoder_layer, num_encoder_layers)

        decoder_layer = DecoderLayer(d_model, num_heads, dim_feedforward, dropout_rate)
        self.decoder = Decoder(decoder_layer, num_decoder_layers)

        # self.set_parameters()

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):

        memory = self.encoder(src, src_mask)
        output = self.decoder(tgt, memory, tgt_mask, memory_mask)

        return output

    def encode(self, src, src_mask):
        return self.encoder(src, src_mask)

    def decode(self, tgt, memory, tgt_mask, memory_mask):
        # print(tgt.shape)
        # print(memory.shape)
        return self.decoder(tgt, memory, tgt_mask, memory_mask)

    def set_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embeddings, self).__init__()

        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * torch.sqrt(torch.tensor(self.d_model))

class PositionalEncoding(nn.Module):
    # "Implement the PE function."
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)

class Generator(nn.Module):
    def __init__(self,d_model=512, vocab_size=37000):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

class Transformer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout_rate=0.1, src_vocab_size=30000, tgt_vocab_size=30000, max_len=500, share_embedding=False):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.src_embedding = Embeddings(src_vocab_size, d_model)
        if share_embedding:
            self.tgt_embedding = self.src_embedding
        else:
            self.tgt_embedding = Embeddings(tgt_vocab_size, d_model)
        self.position_embedding = PositionalEncoding(d_model, dropout=dropout_rate, max_len=max_len)
        self.encoder_decoder = EncoderDecoder(d_model,num_heads,num_encoder_layers,num_decoder_layers,dim_feedforward,dropout_rate)
        self.generator = Generator(d_model, tgt_vocab_size)

        self.set_parameters()

    def forward(self, src, tgt, src_mask, tgt_mask):
        x = self.encoder_decoder(self.position_embedding(self.src_embedding(src)), self.position_embedding(self.tgt_embedding(tgt)), src_mask, tgt_mask, src_mask)
        x = self.generator(x)
        return x

    def encode(self, src, src_mask):
        return self.encoder_decoder.encode(self.position_embedding(self.src_embedding(src)), src_mask)

    def decode(self, tgt, memory, tgt_mask, memory_mask):
        return self.encoder_decoder.decode(self.position_embedding(self.tgt_embedding(tgt)), memory, tgt_mask, memory_mask)

    def set_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

class OptimizerWithWarmUp():
    def __init__(self, model_size, warmup_steps, optimizer):
        self.d_model = model_size
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer
        self._rate = 0
        self._step = 0

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.d_model ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

def get_optimizer_with_warmup(model):
    return OptimizerWithWarmUp(model_size=model.d_model, warmup_steps=4000,
                               optimizer=torch.optim.Adam(model.parameters(), lr=0, betas=(0.9,0.98), eps=1e-9))


def clone_modules(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# 得到的都是True和False的矩阵
def get_padding_mask(target, pad):

    mask = (target != pad).unsqueeze(-2) # [batch_size, 1, tgt_len]  0 means padding

    return mask

# 返回上三角为0的矩阵，其余元素为1的矩阵,工具函数
def get_sequence_mask(target_len):

    ones = torch.ones(target_len, target_len, dtype=torch.uint8)

    mask = torch.triu(ones, diagonal=1) # [tgt_len, tgt_len]

    return 1 - mask # lower triangular matrix: 1 means not masked, 0 means masked

def get_tgt_mask(padding_mask, tgt_len):
    # padding_mask: list, len(list) = S
    if isinstance(padding_mask, list):
        padding_mask = torch.tensor(padding_mask).unsqueeze(0)
    seq_mask = get_sequence_mask(tgt_len)
    tgt_mask = padding_mask & seq_mask
    return tgt_mask


# 添加mask构造，返回上三角为-inf，其余元素为0的矩阵
def generate_square_subsequent_mask(sz, DEVICE):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
