# %%
import json
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import DatasetDict, load_dataset, load_metric
from glob import glob
from tqdm import tqdm

# %%
def audio_to_array(batch: DatasetDict) -> DatasetDict:
    audio_array, sampling_rate = torchaudio.load(batch['path'])
    resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16_000)
    batch['audio'] = resampler(audio_array).squeeze().numpy()
    return batch

# %%
PATH = 'indonesian-nlp/wav2vec2-large-xlsr-indonesian'

processor = Wav2Vec2Processor.from_pretrained(PATH)
model = Wav2Vec2ForCTC.from_pretrained(PATH)
model.to('cuda')

prefix_path = '.\\malay_youtube'

label_paths = glob(pathname=f'{prefix_path}\\*.txt')
audio_paths = glob(pathname=f'{prefix_path}\\*.wav')

assert len(label_paths) == len(audio_paths), 'Number of labels and audios must be equal'

dataset = []

for l, a in tqdm(zip(label_paths, audio_paths)):
    with open(l) as f:
        dataset.append({
            'label': f.readlines(),
            'path': a
            })

json_path = f'{prefix_path}\\malay_youtube.json'

# %%
with open(json_path, 'w') as f:
    for d in dataset:
        json.dump(d, f)

# %%
inference_dataset = load_dataset('json', data_files=json_path)

# %%
inference_dataset = inference_dataset.map(audio_to_array)

# %%
def evaluate(batch: DatasetDict) -> DatasetDict:
    inputs = processor(batch['audio'], sampling_rate=16_000, return_tensors='pt', padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values.to('cuda'), attention_mask=inputs.attention_mask.to('cuda')).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch['pred'] = processor.batch_decode(pred_ids)
    return batch

# %%
results = inference_dataset.map(evaluate)

# %%
metric = load_metric('wer')
print(f'WER: {metric.compute(predictions=results["train"]["pred"], references=results["train"]["label"])}')

# %%
results['train'].remove_columns('audio').to_json('eval_results.json')