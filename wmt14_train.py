import random
import numpy as np

from tokenizers import ByteLevelBPETokenizer, Tokenizer
from tqdm.auto import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import transformer

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
tokenizer = Tokenizer.from_file("de_en_tokenizer/de_en.json")

tokenizer.enable_truncation(max_length=200)
tokenizer.enable_padding(pad_token='<pad>')

################ Prepare Dataset and DataLoader ##################

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

train_src_paths = [str(x) for x in Path('./de_en_data_s').glob('train*de.txt')]
train_tgt_paths = [str(x) for x in Path('./de_en_data_s').glob('train*en.txt')]
valid_src_paths = [str(x) for x in Path('./de_en_data_s').glob('valid*de.txt')]
valid_tgt_paths = [str(x) for x in Path('./de_en_data_s').glob('valid*en.txt')]


tr_set = TransformerDataset(train_src_paths, train_tgt_paths, tokenizer)
val_set = TransformerDataset(valid_src_paths, valid_tgt_paths, tokenizer)

batch_size = 30
train_loader = DataLoader(tr_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)

################ Training ##################
vocab_size = tokenizer.get_vocab_size()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

model_path = './models/model.ckpt'

n_epochs = 100
early_stop = 10

d_model = 512
#max_len = 202

model = transformer.Transformer(d_model=d_model,src_vocab_size=vocab_size, tgt_vocab_size=vocab_size,share_embedding=True).to(device)
# model.load_state_dict(torch.load(model_path))

optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9,0.98), eps=1e-9)
optimizer = transformer.OptimizerWithWarmUp(d_model,8000, optimizer)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

epoch = 0
early_stop_cnt = 0
min_loss = 1000000

while(epoch < n_epochs and early_stop_cnt < early_stop):

    # ---------------- training loop -----------------
    model.train()

    train_loss = []
    train_acc = []
    try:
        with tqdm(train_loader) as t:
            for src, tgt, labels in t:
                # tgt [B, S]  tgt_mask [B, S]
                logits = model(src['data'].to(device), tgt['data'].to(device), src['mask'].to(device), tgt['mask'].to(device) )

                # labels = labels.unsqueeze(-1)
                loss = criterion(logits.view(-1,logits.shape[-1]), labels.view(-1).to(device))

                optimizer.zero_grad()

                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
                # update parameters
                optimizer.step()

                acc = (torch.argmax(logits, dim=-1) == labels.to(device)).float().mean()
                train_loss.append(loss.item())
                train_acc.append(acc)
    except KeyboardInterrupt:
        t.close()
        raise
    t.close()

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_acc) / len(train_acc)

    # print(f"[Train | {epoch + 1: 03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    # ---------------- validating loop -----------------
    model.eval()

    val_loss = []
    val_acc = []

    try:
        with tqdm(val_loader) as t:
            for src, tgt, labels in t:
                logits = model(src['data'].to(device), tgt['data'].to(device), src['mask'].to(device), tgt['mask'].to(device))

            loss = criterion(logits.view(-1,logits.shape[-1]), labels.view(-1).to(device))

            acc = (torch.argmax(logits, dim=-1) == labels.to(device)).float().mean()
            val_loss.append(loss.item())
            val_acc.append(acc)
    except KeyboardInterrupt:
        t.close()
        raise
    t.close()

    val_loss = sum(val_loss) / len(val_loss)
    val_acc = sum(val_acc) / len(val_acc)

    # print(f"[Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {val_loss:.5f}, acc = {val_acc:.5f}")
    print(f"[Train | {epoch + 1: 03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f} || [Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {val_loss:.5f}, acc = {val_acc:.5f}")

    if val_loss < min_loss:
        min_loss = val_loss
        torch.save(model.state_dict(), model_path)
        print(f"saving model with loss {val_loss:3.6f}")
        early_stop_cnt = 0

    else:
        early_stop_cnt += 1

    epoch += 1


