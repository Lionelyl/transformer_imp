# 分词工具
from torchtext.data.utils import get_tokenizer
# 构建数据集的词典的工具
from torchtext.vocab import build_vocab_from_iterator
# Multi30k数据集
# 常用的机器翻译数据集
from torchtext.datasets import Multi30k
from typing import Iterable, List
from transformer import Transformer
from transformer import get_padding_mask
from transformer import get_tgt_mask
from transformer import OptimizerWithWarmUp
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from nltk.translate.bleu_score import sentence_bleu


# 用于产生build_vocab_from_iterator中的迭代器
def yield_tokens(data_iter: Iterable, language: str, src_lang: str, tgt_lang: str) -> List[str]:
    language_index = {src_lang: 0, tgt_lang: 1}
    # 迭代数据集中的每一行
    for data_sample in data_iter:
        # 产生这一行的分词
        yield token_transform[language](data_sample[language_index[language]])


# 建立词典
def build_vocab(src_lang, tgt_lang, vocab_trans, spe_sym, unk):
    for ln in [src_lang, tgt_lang]:
        train_iter = Multi30k(split='train', language_pair=(src_lang, tgt_lang))
        vocab_trans[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln, src_lang, tgt_lang),
                                                    min_freq=1,
                                                    specials=spe_sym,
                                                    special_first=True)
    # 后面翻译过程中要是出现词典中没见过的词语，一律判定为UNK
    for ln in [src_lang, tgt_lang]:
        vocab_trans[ln].set_default_index(unk)


# 从德语翻译到英语
SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

# 转换器
token_transform = {}
vocab_transform = {}

# get_torkenizer(第一个参数指定使用的分词器，如果没有,按照空格进行分割，language=指定使用的分词格式（哪种语言）)

# 德语分词器
token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
# 英语分词器
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')

'''
定义特殊标记
UNK=unknown 未知词语标记
PAD=padding 序列填充标记
BOS=begin of string(有的也定义为SOS，start of string)
EOS=end of string 一句话的结尾标记
'''
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

build_vocab(SRC_LANGUAGE, TGT_LANGUAGE, vocab_transform, special_symbols, UNK_IDX)


# 转换器聚合器 组合多个转换器对输入进行转换
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


# 在一句话的开头和结尾添加特殊标记
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))


# 组合转换器
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # 先分词，然后获取每个词语在词典中的索引
    # 然后转换为张量
    text_transform[ln] = sequential_transforms(token_transform[ln],
                                               vocab_transform[ln],
                                               tensor_transform)

'''
处理一批次的数据：
1、首先是分词、转换索引、转换张量并添加到一个batch中。
2、然后通过pad_sequence将一个batch内的所有数据统一长度
3、return：[统一的seq_len,batch_size]
'''


def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


# 测试
# print(vocab_transform['en'].get_stoi())
print("创建词典成功！")
# 获取可用设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(0)
# 词典的大小
SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
# 词嵌入维度
EMB_SIZE = 512
# 多头注意力头数
NHEAD = 8
# Transformer隐藏层的维度
FFN_HID_DIM = 2048
# 批大小
BATCH_SIZE = 128
# 编、解码器的层数
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DROPOUT_RATE = 0.1

# 实例化一个自定义的Transformer
transformer = Transformer(EMB_SIZE, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS,
                          FFN_HID_DIM, DROPOUT_RATE, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE)

# 如果GPU可用放在GPU上
transformer = transformer.to(DEVICE)
# 损失函数
# padding的不是真正的词语，要忽略掉 ignore_index=PAD_IDX
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
# Adam优化算法
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
optimizer = OptimizerWithWarmUp(EMB_SIZE, 4000, optimizer)


def train_epoch(model, optimizer):
    # 开启模型训练模式
    model.train()
    losses = 0
    # 获取一批次的数据
    # 并指定对于这批数据使用我们上面定义的处理方法， 转换为张量
    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    i = 0
    # 训练
    for src, tgt in train_dataloader:
        i += 1
        src = src.t()  # batch first
        tgt = tgt.t()
        # 如果GPU可用放在GPU上
        src = src.to(DEVICE)
        # 省去最后一个结尾特殊标志
        tgt_input = tgt[:, :-1]
        # 获取Mask矩阵
        src_mask = get_padding_mask(src, PAD_IDX)
        src_mask = src_mask.to(DEVICE)
        tgt_padding_mask = get_padding_mask(tgt_input, PAD_IDX)
        tgt_mask = get_tgt_mask(tgt_padding_mask, tgt_input.size(1))
        tgt_mask = tgt_mask.to(DEVICE)
        tgt_input = tgt_input.to(DEVICE)
        tgt = tgt.to(DEVICE)
        # tgt_padding_mask = tgt_padding_mask.to(DEVICE)
        # print("src_mask:", src_mask.shape)
        # print("tgt_mask:", tgt_mask.shape)
        # 获取模型的输出
        # logits是没有经过softmax的模型输出的意思
        logits = model(src, tgt_input, src_mask, tgt_mask)  # [B, L-1, TGT_VOCAB_SIZE]
        # print(logits.shape)
        # 梯度清零
        optimizer.zero_grad()
        # 第一个是开始特殊标记
        # 省去
        tgt_out = tgt[:, 1:]  # [B, L-1]
        # logits.reshape(-1, logits.shape[-1])后维度变为[(seq_len-1)*batch_size,目标语言的词典大小]
        # tgt_out.reshape(-1)后tgt_out的维度是[seq_len-1*batch_size*目标语言的词典大小]
        # tgt_out的最后一个维度是真实标签
        # 而logits的最后一个维度是对于每一个词语的预测
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        # 反向传播
        loss.backward()
        # 梯度更新
        optimizer.step()
        losses += loss.item()
    # 取平均的loss
    return losses / len(train_dataloader)


# 模型验证
def evaluate(model):
    model.eval()
    losses = 0
    val_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    for src, tgt in val_dataloader:
        src = src.t()  # batch first
        tgt = tgt.t()
        src = src.to(DEVICE)

        tgt_input = tgt[:, :-1]
        # 获取Mask矩阵
        src_mask = get_padding_mask(src, PAD_IDX)
        tgt_padding_mask = get_padding_mask(tgt_input, PAD_IDX)
        tgt_mask = get_tgt_mask(tgt_padding_mask, tgt_input.size(1))
        tgt_mask = tgt_mask.to(DEVICE)
        tgt_input = tgt_input.to(DEVICE)
        tgt = tgt.to(DEVICE)
        # 开始训练
        logits = model(src, tgt_input, src_mask, tgt_mask)
        tgt_out = tgt[:, 1:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
    return losses / len(val_dataloader)


NUM_EPOCHS = 15
# 迭代训练和验证
for epoch in range(1, NUM_EPOCHS + 1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer)
    end_time = timer()
    val_loss = evaluate(transformer)
    print((
        f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))


# 让训练完的模型产生翻译字符串
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long)
    for i in range(max_len - 1):
        memory = memory.to(DEVICE)
        ys = ys.to("cpu")
        tgt_padding_mask = get_padding_mask(ys, PAD_IDX)
        ys = ys.to(DEVICE)
        tgt_mask = get_tgt_mask(tgt_padding_mask, ys.size(1))
        tgt_mask = tgt_mask.to(DEVICE)
        out = model.decode(ys, memory, tgt_mask, src_mask)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        if next_word == EOS_IDX:
            break
    return ys


# 翻译
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src = src.t()  # batch first
    src_mask = (torch.ones(1, num_tokens))  # 全1表示没有mask的内容
    tgt_tokens = greedy_decode(
        model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>",
                                                                                                         "").replace(
        "<eos>", "")


# 计算Blue Score
# 首先读取测试集中的德语数据
de_sent = []
with open('.data/multi30k/test.de', encoding='utf-8') as f:
    for line in f:
        lst_temp = [line.strip('\n')]
        de_sent.append(lst_temp)
en_candidate_list = []  # 将翻译之后的英文存储在该列表中
for de in de_sent:
    en_candidate_list.append(translate(transformer, de[0]))
print("翻译结束")
print(en_candidate_list[:5])
en_candidate_list_tokenize = []  # 将翻译之后的英文分词后结果
for sents in en_candidate_list:
    res = token_transform[TGT_LANGUAGE](sents)
    en_candidate_list_tokenize.append(res)
print(en_candidate_list_tokenize[:5])

# 读取测试集中英语数据存储在reference
en_sent = []
with open('.data/multi30k/test.en', encoding='utf-8') as f:
    for line in f:
        lst_temp = [line.strip('\n')]
        en_sent.append(lst_temp)

print(len(en_sent))
en_sent_tokenize = []
for sents in en_sent:
    res = token_transform[TGT_LANGUAGE](sents[0])
    en_sent_tokenize.append(res)

# 计算Blue评分
i = 0
total_blue_score = 0
for candidate in en_candidate_list_tokenize:
    reference = [en_sent_tokenize[i]]
    print("reference为：", reference)
    print("candidate为：", candidate)
    score = sentence_bleu(reference, candidate) * 100
    total_blue_score += score
    i += 1
print("测试集中平均的Blue score为：", total_blue_score / i)
