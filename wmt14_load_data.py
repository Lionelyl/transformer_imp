import datasets
from tqdm.auto import tqdm
import os

################ Load Dataset ##################

de_en = datasets.load_dataset('wmt14', 'de-en')

################ Save Tokenizer ##################
en_txt = []
de_txt = []
file_count = 0
max_file = 3
# os.mkdir('de_en_data_s')


for sample in tqdm(de_en['train']['translation']):
    de = sample['de']
    en = sample['en']
    de_txt.append(de)
    en_txt.append(en)
    if file_count < max_file:
        if len(en_txt) >= 10_000:
            with open(f'de_en_data_s/train_{file_count}_de.txt', 'w', encoding='utf-8') as fd:
                fd.write('\n'.join(de_txt))
            de_txt = []
            with open(f'de_en_data_s/train_{file_count}_en.txt', 'w', encoding='utf-8') as fd:
                fd.write('\n'.join(en_txt))
            en_txt = []
            file_count += 1
    else:
        break

en_txt = []
de_txt = []
for sample in tqdm(de_en['validation']['translation']):
    de = sample['de']
    en = sample['en']
    de_txt.append(de)
    en_txt.append(en)
    if len(en_txt) >= 3_000:
        with open(f'de_en_data_s/valid_de.txt', 'w', encoding='utf-8') as fd:
            fd.write('\n'.join(de_txt))
        de_txt = []
        with open(f'de_en_data_s/valid_en.txt', 'w', encoding='utf-8') as fd:
            fd.write('\n'.join(en_txt))
        en_txt = []
        break
