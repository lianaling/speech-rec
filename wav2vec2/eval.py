# %%

from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets
import argparse
from tqdm.auto import tqdm
from torchmetrics import WordErrorRate


# parser = argparse.ArgumentParser(description='Predict using Wav2Vec 2.0 model')
# parser.add_argument('--dataset_path', type=str, help='Dataset for inference')
# parser.add_argument('--lang', type=str, help='Dataset language')
# args = parser.parse_args()

PATH = 'facebook/wav2vec2-base-960h'

pipe = pipeline('automatic-speech-recognition', model=PATH, device=0)
dataset = datasets.load_dataset('librispeech_asr', name='clean', split='test', data_files={'test': 'http://www.openslr.org/resources/12/test-clean.tar.gz'})
# dataset = datasets.load_from_disk("/c/Users/liana/.cache/huggingface/datasets/downloads/extracted/9a7f02367a1870a7e4513074f89aaf74f637e8125390879eb3f37ea50bf83814/")

# print(pipe(KeyDataset(dataset, 'text')))

targets = []
preds = []

# for out in tqdm(pipe(KeyDataset(dataset, 'file'))):
#     preds.append(out)

[targets.append(tar) for tar in tqdm(dataset['text'])]

# %%

for tar in tqdm(pipe(KeyDataset(dataset, 'file'))):
    preds.append(tar['text'])

# %%
metric = WordErrorRate()
print(f'WER: {metric(preds, targets) * 100}')
