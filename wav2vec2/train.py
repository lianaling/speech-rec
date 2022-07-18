# %% [markdown]
# # References
# [Fine-Tune Wav2Vec2 for English ASR with 🤗 Transformers](https://huggingface.co/blog/fine-tune-wav2vec2-english)
# [How to train a new language model from scratch using Transformers and Tokenizers](https://huggingface.co/blog/how-to-train)

# %%
import datasets as hfds
from torch import flatten
import transformers
import tokenizers
import os
import json
import tqdm
from typing import Dict, List, Union, Tuple

# %%
def extract_all_chars(batch: hfds.DatasetDict) -> Dict[str, List[str]]:
    all_text = " ".join(batch['text'])
    vocab = [set(all_text)]
    return {'vocab': [vocab], 'all_text': [all_text]}

# %%
def downsize_dataset(dataset: hfds.DatasetDict, num_rows: int) -> hfds.DatasetDict:
    return (dataset.shuffle(seed=42)).select([x for x in range(num_rows)])

# %%
def prepare_tokenizer_dataset(*datasets: Union[hfds.DatasetDict, Tuple[hfds.DatasetDict]]) -> hfds.DatasetDict:
    to_combine = []
    for d in datasets:
        # Keep only the 'text' column
        # Remove 'text' column after extracting all characters
        to_combine.append(d.remove_columns([col for col in d.column_names if col != 'text']).map(extract_all_chars, keep_in_memory=True, remove_columns='text'))

    return hfds.concatenate_datasets([*to_combine])
# %%
def __flatten(list: List[List]) -> List:
    '''Flattens an n-dimensional list with only one element in each nest except for the last list to a one-dimensional list
    Applicable only to lists like this: x = [[[[0, 1, 2]]]]
    '''
    item = list[0]
    if not isinstance(item, type(list)):
        return list
    else:
        list = __flatten(item)

    return list

# %%
def generate_vocab_dict(tokenizer_dataset: hfds.DatasetDict) -> Dict[str, int]:
    tokenizer_dataset = __flatten(tokenizer_dataset['vocab'])
    vocabs = set(tokenizer_dataset)
    return {v: k for k, v in enumerate(vocabs)}

#%%
def prepare_dataset(batch: hfds.DatasetDict) -> List[Dict]:
    audio = batch['audio']

    batch['input_values'] = processor(audio['array'], sampling_rate=audio['sampling_rate']).input_values[0]

    with processor.as_target_processor():
        batch['labels'] = processor(batch['text']).input_ids

    return batch
# %%
# Prepare dataset
train_dataset = hfds.load_dataset('librispeech_asr', name='clean', split='train.100', data_files={'test': 'http://www.openslr.org/resources/12/test-clean.tar.gz'})

val_dataset = hfds.load_dataset('librispeech_asr', name='clean', split='validation', data_files={'test': 'http://www.openslr.org/resources/12/test-clean.tar.gz'})

# %%
# Small data subset
train_dataset = downsize_dataset(train_dataset, 1000)
val_dataset = downsize_dataset(val_dataset, 100)
# %%
# Prepare tokenizer dataset
tokenizer_dataset = prepare_tokenizer_dataset(train_dataset, val_dataset)
# %%
# Generate vocab_dict
vocab_dict = generate_vocab_dict(tokenizer_dataset)
# Make whitespace more visible
vocab_dict['|'] = vocab_dict[' ']
# Add unknown and padding tokens
vocab_dict['[UNK]'] = len(vocab_dict)
vocab_dict['[PAD]'] = len(vocab_dict)
# %%
# Save vocab as json file
with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

# %%
tokenizer = transformers.Wav2Vec2CTCTokenizer('vocab.json', unk_token='[UNK]', pad_token='[PAD]', word_delimiter_token='|')
# %%
# Training BPE from scratch

with open('vocab.txt', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

bpe_tokenizer = tokenizers.ByteLevelBPETokenizer()

bpe_tokenizer.train(files='vocab.txt', special_tokens=['[UNK]', '[PAD]', '|'])

bpe_tokenizer.save_model('.', 'from-scratch')
# %%
feature_extractor = transformers.Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)

# %%
processor = transformers.Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# %%
tqdm(train_dataset.map(prepare_dataset, remove_columns=[col for col in train_dataset.column_names if col != 'input_values' or col != 'labels'], num_proc=4))
# %%
