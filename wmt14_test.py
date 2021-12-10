import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import transformer
from nltk.translate import bleu_score

import random
import numpy as np
from tokenizers import Tokenizer
from pathlib import Path
from tqdm.auto import tqdm

class TransformerDataset(Dataset):
    def __init__(self, src_files, tgt_files, tokenizer):
        super(TransformerDataset, self).__init__()

        if tgt_files is not None:
            self.mode = 'train_dev'
        else:
            self.mode = 'test'

        src_txt = []
        for file in tqdm(src_files):
            with open(file, 'r') as fp:
                src_txt += fp.read().split('\n')

        self.src = tokenizer.encode_batch(src_txt)
        if self.mode !='test':
            tgt_txt = []
            for file in tqdm(tgt_files):
                with open(file, 'r') as fp:
                    tgt_txt += fp.read().split('\n')

            self.tgt = tokenizer.encode_batch(tgt_txt)


    def __getitem__(self, id):
        if self.mode == 'test':
            src_ids = self.src[id].ids
            src_mask = self.src[id].attention_mask
            src_mask = torch.tensor(src_mask, dtype=torch.int).unsqueeze(0)
            src = {'data':torch.tensor(src_ids, dtype=torch.int), 'mask':src_mask}
            return src
        else:
            # tgt_sos_padding = torch.ones(len(self.tgt)).type(torch.int) # '<sos>' 1
            # label_eos_padding = tgt_sos_padding + tgt_sos_padding      # '<eos>' 2
            src_ids = self.src[id].ids
            src_mask = self.src[id].attention_mask
            src_mask = torch.tensor(src_mask, dtype=torch.int).unsqueeze(0)
            tgt_ids = [1] + self.tgt[id].ids
            tgt_mask = [1] + self.tgt[id].attention_mask
            tgt_mask = transformer.get_tgt_mask(tgt_mask, len(tgt_mask))

            label_ids = self.tgt[id].ids + [2]
            # label_mask = self.tgt[id].attention_mask + [1]

            src = {'data':torch.tensor(src_ids, dtype=torch.int), 'mask':src_mask}
            tgt = {'data':torch.tensor(tgt_ids, dtype=torch.int), 'mask':tgt_mask}
            label = torch.tensor(label_ids, dtype=torch.int)
            return src, tgt, label

    def __len__(self):
        return len(self.src)

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

################ Prepare Tokenizer ##################
print("################ Prepare Tokenizer ##################")

tokenizer = Tokenizer.from_file("de_en_tokenizer/de_en.json")

tokenizer.enable_truncation(max_length=200)
tokenizer.enable_padding(pad_token='<pad>')

################ Prepare Dataset and DataLoader ##################
print("################ Prepare Dataset and DataLoader ##################")

tt_src_paths = [str(x) for x in Path('./de_en_data_s').glob('valid*de.txt')]
# valid_tgt_paths = [str(x) for x in Path('./de_en_data_s').glob('valid*de.txt')]

tt_set = TransformerDataset(tt_src_paths, None, tokenizer=tokenizer)

batch_size = 1

tt_loader = DataLoader(tt_set, batch_size=batch_size)

################ Testing ##################
print("################ Testing ##################")

vocab_size = tokenizer.get_vocab_size()
d_model = 512
max_len = 202

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_path = './models/model.ckpt'
model = transformer.Transformer(d_model=d_model,src_vocab_size=vocab_size, tgt_vocab_size=vocab_size, share_embedding=True).to(device)
# model.load_state_dict(torch.load(model_path))



# <sos> 1, <eos> 2
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
sos_id = 1
eos_id = 2
i = 0
with tqdm(tt_loader) as t:
    scores = []
    for data in t:
        src = data['data'].to(device)
        # print(tokenizer.decode(data['data'].squeeze().tolist()))
        src_mask = data['mask'].to(device)
        out = greed_decode(model, src, src_mask, max_len, sos_id, eos_id)
        # print(tokenizer.decode(out.squeeze().tolist()))
        reference = tokenizer.decode(data['data'].squeeze().tolist())
        reference = [tokenizer.encode(reference).tokens]
        print(reference)
        candidate = [tokenizer.encode(tokenizer.decode(out.squeeze().tolist())).tokens]
        print(candidate)
        score = bleu_score.corpus_bleu(reference, candidate)
        print(score)
        scores.append(score)
        if (i>2):
            break
        i += 1
    average_bleu = sum(scores) / len(scores)
    print(f"BLEU scores : {average_bleu:3.6f}")