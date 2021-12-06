import torch
import torch.nn as nn
import torchtext
from torchtext import data
from torchtext.data.utils import get_tokenizer
import tqdm
TEXT = torchtext.data.Field(tokenize=get_tokenizer("spacy"),
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)

train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)

TEXT.build_vocab(train_txt)

def batchify(data, bsize):

    data = TEXT.numericalize([data.examples[0].text])
    print(data)
    nbatch = data.size(0) // bsize

    data = data.narrow(0, 0, nbatch * bsize)

    data = data.view(bsize, -1)

    return data

batch_size = 20

eval_batch_size = 10

train_data = batchify(train_txt, batch_size)
val_data = batchify(val_txt, eval_batch_size)
test_data = batchify(test_txt, eval_batch_size)

# 令子长度允许的最大值bptt为35
bptt = 35

def get_batch(source, i):
    """用于获得每个批次合理大小的源数据和目标数据.
       参数source是通过batchify得到的train_data/val_data/test_data.
       i是具体的批次次数.
    """

    # 首先我们确定句子长度, 它将是在bptt和len(source) - 1 - i中最小值
    # 实质上, 前面的批次中都会是bptt的值, 只不过最后一个批次中, 句子长度
    # 可能不够bptt的35个, 因此会变为len(source) - 1 - i的值.
    seq_len = min(bptt, len(source) - 1 - i)

    # 语言模型训练的源数据的第i批数据将是batchify的结果的切片[i:i+seq_len]
    data = source[i:i+seq_len]

    # 根据语言模型训练的语料规定, 它的目标数据是源数据向后移动一位
    # 因为最后目标数据的切片会越界, 因此使用view(-1)来保证形状正常.
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

source = test_data
i = 1

get_batch(source, i)
