import datasets
from tokenizers import ByteLevelBPETokenizer, Tokenizer
from tqdm.auto import tqdm
from pathlib import Path
import os

# ################ Load Dataset ##################
#
# de_en = datasets.load_dataset('wmt14', 'de-en')
#
# ################ Save Tokenizer ##################
# en_txt = []
# de_txt = []
# file_count = 0
# os.mkdir('de_en_data')
# for sample in tqdm(de_en['train']['translation']):
#     de = sample['de']
#     en = sample['en']
#     de_txt.append(de)
#     en_txt.append(en)
#     if len(en_txt) >= 10_000:
#         with open(f'de_en_data/de_en_{file_count}_de.txt', 'w', encoding='utf-8') as fd:
#             fd.write('\n'.join(de_txt))
#         de_txt = []
#         with open(f'de_en_data/de_en_{file_count}_en.txt', 'w', encoding='utf-8') as fd:
#             fd.write('\n'.join(en_txt))
#         en_txt = []
#         file_count += 1
#
# with open(f'de_en_{file_count}_de.txt', 'w', encoding='utf-8') as fd:
#     fd.write('\n'.join(de_txt))
#     de_txt = []
# with open(f'de_en_{file_count}_en.txt', 'w', encoding='utf-8') as fd:
#     fd.write('\n'.join(en_txt))
#     en_txt = []

################ Prepare Tokenizer ##################

paths = [str(x) for x in Path('./de_en_data/').glob('*.txt')]
paths.sort()

# tokenizer = ByteLevelBPETokenizer()
#
# tokenizer.train(
#     files=paths[:100],
#     vocab_size=37000,
#     min_frequency=2,
#     show_progress=True,
#     special_tokens=['<pad>', '<sos>', '<eos>', '<unk>', '<mask>']
# )
#
# os.mkdir('de_en_tokenizer')
# tokenizer.save('de_en_tokenizer/de_en.json')

tokenizer = Tokenizer.from_file("de_en_tokenizer/de_en.json")

tokenizer.enable_truncation(max_length=128)
tokenizer.enable_padding(pad_token='<pad>')

################ Prepare Dataset and DataLoader ##################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import transformer

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

        self.src = tokenizer.encode_batch(src_txt[:1000])
        if self.mode !='test':
            tgt_txt = []
            for file in tqdm(tgt_files):
                with open(file, 'r') as fp:
                    tgt_txt += fp.read().split('\n')

            self.tgt = tokenizer.encode_batch(tgt_txt[:1000])


    def __getitem__(self, id):
        if self.mode == 'test':
            return {'data': self.src[id].ids, 'mask': self.src[id].attention_mask}
        else:
            # tgt_sos_padding = torch.ones(len(self.tgt)).type(torch.int) # '<sos>' 1
            # label_eos_padding = tgt_sos_padding + tgt_sos_padding      # '<eos>' 2
            src_ids = self.src[id].ids
            src_mask = self.src[id].attention_mask
            src_mask = torch.tensor(src_mask, dtype=int).unsqueeze(0)
            tgt_ids = [1] + self.tgt[id].ids
            tgt_mask = [1] + self.tgt[id].attention_mask
            tgt_mask = transformer.get_tgt_mask(tgt_mask, len(tgt_mask))

            label_ids = self.tgt[id].ids + [2]
            # label_mask = self.tgt[id].attention_mask + [1]

            src = {'data':torch.tensor(src_ids, dtype=int), 'mask':src_mask}
            tgt = {'data':torch.tensor(tgt_ids, dtype=int), 'mask':tgt_mask}
            label = torch.tensor(label_ids, dtype=int)
            return src, tgt, label

    def __len__(self):
        return len(self.src)

src_paths = [str(x) for x in Path('./de_en_data').glob('*en.txt')]
tgt_paths = [str(x) for x in Path('./de_en_data').glob('*de.txt')]
#########test small dataset###########

val_rate = 0.2
max_datasize = 10
edge = round(10*(1-val_rate))

# tr_set = TransformerDataset(['src_txt_tr.txt'], ['tgt_txt_tr.txt'], tokenizer)
# val_set = TransformerDataset(['src_txt_val.txt'], ['tgt_txt_val.txt'], tokenizer)
tr_set = TransformerDataset(src_paths[:1], tgt_paths[:1], tokenizer)
val_set = TransformerDataset(src_paths[1:2], tgt_paths[1:2], tokenizer)

train_loader = DataLoader(tr_set, 10, shuffle=True)
val_loader = DataLoader(val_set, 10)

################ Training ##################
vocab_size = tokenizer.get_vocab_size()

n_epochs = 1000
early_stop = 50

d_model = 512
max_len = 135

# model = EncoderDecoder(d_model,vocab_size, max_len)
model = transformer.Transformer(d_model=d_model,vocab_size=vocab_size, max_len=max_len)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

epoch = 0
early_stop_cnt = 0
min_loss = 1000000

while(epoch < n_epochs and early_stop_cnt < early_stop):

    model.train()

    train_loss = []
    train_acc = []
    try:
        with tqdm(train_loader) as t:
            for src, tgt, labels in t:
                # src [B, L]  src_mask [B, L]
                # tgt [B, S]  tgt_mask [B, S]
                logits = model(src['data'], tgt['data'], src['mask'], tgt['mask'] )

                 # labels = labels.unsqueeze(-1)
                loss = criterion(logits.view(-1,logits.shape[-1]), labels.view(-1))

                optimizer.zero_grad()

                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
                # update parameters
                optimizer.step()

                acc = (torch.argmax(logits, dim=-1) == labels).float().mean()
                train_loss.append(loss.item())
                train_acc.append(acc)
    except KeyboardInterrupt:
        t.close()
        raise
    t.close()

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_acc) / len(train_acc)

    print(f"[Train | {epoch + 1: 03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    model.eval()

    val_loss = []
    val_acc = []

    try:
        with tqdm(val_loader) as t:
            for src, tgt, labels in t:
                logits = model(src['data'], tgt['data'], src['mask'], tgt['mask'] )

            loss = criterion(logits.view(-1,logits.shape[-1]), labels.view(-1))

            acc = (torch.argmax(logits, dim=-1) == labels).float().mean()
            val_loss.append(loss.item())
            val_acc.append(acc)
    except KeyboardInterrupt:
        t.close()
        raise
    t.close()

    val_loss = sum(val_loss) / len(val_loss)
    val_acc = sum(val_acc) / len(val_acc)

    print(f"Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {val_loss:.5f}, acc = {val_acc:.5f}")

    if val_loss < min_loss:
        min_loss = val_loss
        early_stop_cnt = 0

    else:
        early_stop_cnt += 1

    epoch += 1

################ Load Dataset ##################


