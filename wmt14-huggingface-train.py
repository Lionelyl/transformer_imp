import random
import numpy as np
import torch
import torch.nn as nn

import transformer
import datasets
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

train_set= datasets.load_dataset('wmt14', 'de-en', split='train').flatten()
valid_set = datasets.load_dataset('wmt14', 'de-en', split='validation').flatten()
test_set = datasets.load_dataset('wmt14', 'de-en', split='test').flatten()

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

batch_size = 32
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(valid_set, batch_size=batch_size, collate_fn=collate_fn)
test_dataloader = DataLoader(test_set, batch_size=batch_size, collate_fn=collate_fn)


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


################ Training ##################
vocab_size = tokenizer.vocab_size

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

model_path = './models/model.ckpt'

n_epochs = 100
early_stop = 10

d_model = 512


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
    with tqdm(val_dataloader) as t:
        for data in t:
            # tgt [B, S]  tgt_mask [B, S]
            src_mask = data['src_mask'].unsqueeze(1)
            tgt_mask = transformer.get_tgt_mask(data['tgt_mask'].unsqueeze(1), data['tgt_mask'].shape[1])

            logits = model(data['src_ids'].to(device), data['tgt_ids'].to(device), src_mask.to(device), tgt_mask.to(device) )

            # labels = labels.unsqueeze(-1)
            labels = torch.cat([data['tgt_ids'][:,1:], data['tgt_ids'][:,-1].unsqueeze(-1)], dim=-1)
            loss = criterion(logits.view(-1,logits.shape[-1]), labels.view(-1).to(device))

            optimizer.zero_grad()

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            # update parameters
            optimizer.step()

            acc = (torch.argmax(logits, dim=-1) == labels.to(device)).float().mean()
            train_loss.append(loss.item())
            train_acc.append(acc)

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_acc) / len(train_acc)

    # print(f"[Train | {epoch + 1: 03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    # ---------------- validating loop -----------------
    model.eval()

    val_loss = []
    val_acc = []


    with tqdm(val_dataloader) as t:
        for data in t:
            # tgt [B, S]  tgt_mask [B, S]
            src_mask = data['src_mask'].unsqueeze(1)
            tgt_mask = transformer.get_tgt_mask(data['tgt_mask'].unsqueeze(1), data['tgt_mask'].shape[1])
            logits = model(data['src_ids'].to(device), data['tgt_ids'].to(device), src_mask.to(device), tgt_mask.to(device) )

            labels = torch.cat([data['tgt_ids'][:,1:], data['tgt_ids'][:,-1].unsqueeze(-1)], dim=-1)
            loss = criterion(logits.view(-1,logits.shape[-1]), labels.view(-1).to(device))

            acc = (torch.argmax(logits, dim=-1) == labels.to(device)).float().mean()
            val_loss.append(loss.item())
            val_acc.append(acc)

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


