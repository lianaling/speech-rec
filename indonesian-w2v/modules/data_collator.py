import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from transformers import Wav2Vec2Processor

@dataclass
class DataCollatorCTCWithPadding:
    '''Pad to maximum batch length.'''
    processor: Wav2Vec2Processor
    padding: bool | str
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, List[int] | torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_features = [{'input_values': feature['input_values']} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt'
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors='pt'
            )

        labels = labels_batch['input_ids'].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch['labels'] = labels

        return batch