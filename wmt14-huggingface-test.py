import random
import numpy as np
import torch
import torch.nn as nn

import transformer
import datasets
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

################ Set seed ##################
seed = 73
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


################ Prepare Test Data ##################
test_set = datasets.load_dataset('wmt14', 'de-en', split='test').flatten()
# print(test_set.features)
tokenizer = AutoTokenizer.from_pretrained("google/bert2bert_L-24_wmt_de_en", pad_token="<pad>", eos_token="</s>", bos_token="<s>")

def collate_fn(batch):
    src_sentences = [item['translation.de'] for item in batch]
    tgt_sentences = [item['translation.en'] for item in batch]
    src = tokenizer(src_sentences, padding='longest', truncation=True, max_length=512, return_tensors="pt",add_special_tokens=False)
    tgt = tokenizer(tgt_sentences, padding='longest', truncation=True, max_length=512, return_tensors="pt",)
    return {'src_ids': src['input_ids'],
            'src_mask': src['attention_mask'],
            'tgt_ids': tgt['input_ids'],
            'tgt_mask': tgt['attention_mask'],
            'src_sentences': src_sentences,
            'tgt_sentences': tgt_sentences
            }

batch_size = 1
test_dataloader = DataLoader(test_set, batch_size=batch_size, collate_fn=collate_fn)


################ Testing ##################
print("################ Testing ##################")

vocab_size = tokenizer.vocab_size
d_model = 512

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_path = './models/model.ckpt'
model = transformer.Transformer(d_model=d_model,src_vocab_size=vocab_size, tgt_vocab_size=vocab_size, share_embedding=True).to(device)
# model.load_state_dict(torch.load(model_path))


# <sos> 101, <eos> 102
def greed_decode(model, src, src_mask, max_len, sos_id, eos_id):
    # src [1, s]
    # src_mask [1, 1, s]

    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask).to(device)

    ys = torch.ones(1,1).fill_(sos_id).type(torch.int).to(device)
    for i in range(max_len-1):
        ys_mask = transformer.get_sequence_mask(ys.shape[1]).unsqueeze(0).to(device)
        out = model.decode(ys, memory, ys_mask, src_mask)
        out = model.generator(out)
        out = torch.argmax(out,dim=-1)
        next_word = out[:,-1]
        ys = torch.cat([ys,next_word.unsqueeze(0)], dim=1)
        if next_word == eos_id:
            break
    return ys



print("################ Greedy Decode ##################")
sos_id = 101
eos_id = 102
i = 0
# metric = datasets.load_metric("bleu")
max_len = 202

with tqdm(test_dataloader) as t:
    scores = []
    for data in t:
        src = data['src_ids'].to(device)
        src_mask = data['src_mask'].unsqueeze(1).to(device)

        out = greed_decode(model, src, src_mask, max_len, sos_id, eos_id)
        print(out, out.shape)
        src_sentence = data['src_sentences']
        tgt_sentence = data['tgt_sentences']
        print(src_sentence)
        print(tgt_sentence)
        print(tokenizer.decode(out.squeeze().tolist()))

        # print(reference)
        # candidate = [tokenizer.encode(tokenizer.decode(out.squeeze().tolist())).tokens]
        # print(candidate)
        # score = bleu_score.corpus_bleu(reference, candidate)
        # print(score)
        # scores.append(score)
        if (i>2):
            break
        i += 1
    # average_bleu = sum(scores) / len(scores)
    # print(f"BLEU scores : {average_bleu:3.6f}")