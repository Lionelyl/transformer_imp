from tokenizers import ByteLevelBPETokenizer, Tokenizer
from pathlib import Path
import os
################ Prepare Tokenizer ##################

# paths = [str(x) for x in Path('./de_en_data_s/').glob('train*.txt')]
# paths.sort()
# print(paths)
#
# tokenizer = ByteLevelBPETokenizer()
#
# tokenizer.train(
#     files=paths,
#     vocab_size=37000,
#     min_frequency=2,
#     show_progress=True,
#     special_tokens=['<pad>', '<sos>', '<eos>', '<unk>', '<mask>']
# )
#
# # os.mkdir('de_en_tokenizer')
# tokenizer.save('de_en_tokenizer/de_en.json')

tokenizer = Tokenizer.from_file("de_en_tokenizer/de_en.json")

#
# tokenizer.enable_truncation(max_length=200)
# tokenizer.enable_padding(pad_token='<pad>')
# print(tokenizer.get_vocab_size())