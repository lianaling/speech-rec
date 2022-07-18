# To prepare iium dataset manifest
# Format: json
# {"path": audio_path.wav, "label": ['This', 'is', 'a', 'label']}
# {"path": audio_path.wav, "label": ['This', 'is', 'a', 'label']}
from glob import glob
import json

DIR_ROOT = '.'

audio_paths = glob(f'{DIR_ROOT}\\audio-iium\\*.wav')
labels = []

with open(f'{DIR_ROOT}\\shuffled-iium.json') as f:
    labels = f.readline()
    labels = labels[1:-2].split('",')
    labels = list(map(lambda l: l.strip('"'), labels))

# labels = labels[2:] # HACK: missing first two wav files so discard first two transcripts

print(labels[0])
print(type(labels[0]))

with open('tmp.txt', 'w') as f:
    for l in labels:
        f.writelines(l)

assert len(audio_paths) == len(labels), f'Number of audio files {len(audio_paths)} and labels {len(labels)} should be equal'

with open(f'{DIR_ROOT}\\manifest-iium.json', 'w') as f:
    for a, l in zip(audio_paths, labels):
        json.dump()