"""
Data preparation pipeline for NeuralFlex-MoE, enhanced for multi-modal "Any-to-Text" training.
"""

from datasets import load_from_disk, concatenate_datasets
from transformers import AutoTokenizer, AutoImageProcessor, Wav2Vec2FeatureExtractor
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional
import os
import logging

logger = logging.getLogger(__name__)

class MultiModalDataset(Dataset):
    """
    A dataset class that wraps a Hugging Face dataset to handle text, image, and audio modalities.
    """
    def __init__(self, hf_dataset, tokenizer, image_processor, audio_processor, text_col="text", image_col="image", audio_col="audio"):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.audio_processor = audio_processor
        self.text_col = text_col
        self.image_col = image_col
        self.audio_col = audio_col
        self.column_names = hf_dataset.column_names

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        
        # Process text
        text = item.get(self.text_col, "")
        # For vision/audio tasks, the text is often a caption/transcript we want the model to generate
        # We formulate this as an instruction-following task
        if self.image_col in item and item[self.image_col] is not None:
            text = f"Describe the following image: {text}"
        elif self.audio_col in item and item[self.audio_col] is not None:
            text = f"Transcribe the following audio: {text}"
            
        tokenized_text = self.tokenizer(text, truncation=True, padding="max_length", return_tensors="pt")

        # Process image
        pixel_values = None
        if self.image_col in self.column_names and item.get(self.image_col) is not None:
            try:
                # The image processor expects a PIL image
                image = item[self.image_col].convert("RGB")
                pixel_values = self.image_processor(image, return_tensors="pt")['pixel_values'].squeeze(0)
            except Exception as e:
                logger.warning(f"Could not process image at index {idx}: {e}")

        # Process audio
        audio_values = None
        if self.audio_col in self.column_names and item.get(self.audio_col) is not None:
            try:
                # The audio processor expects a raw waveform and sampling rate
                audio_data = item[self.audio_col]
                waveform = audio_data['array']
                sampling_rate = audio_data['sampling_rate']
                audio_values = self.audio_processor(waveform, sampling_rate=sampling_rate, return_tensors="pt")['input_values'].squeeze(0)
            except Exception as e:
                logger.warning(f"Could not process audio at index {idx}: {e}")

        return {
            "input_ids": tokenized_text["input_ids"].squeeze(0),
            "attention_mask": tokenized_text["attention_mask"].squeeze(0),
            "pixel_values": pixel_values,
            "audio_values": audio_values,
        }

@dataclass
class DataCollatorForMultiModal:
    """
    A data collator that intelligently batches text, image, and audio data.
    """
    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch = {}
        
        # Handle text
        batch["input_ids"] = torch.stack([f["input_ids"] for f in features])
        batch["attention_mask"] = torch.stack([f["attention_mask"] for f in features])
        
        # Handle images - only include if at least one sample in the batch has an image
        if any(f["pixel_values"] is not None for f in features):
            # If a sample has no image, we create a tensor of zeros
            default_image = torch.zeros_like(next(f["pixel_values"] for f in features if f["pixel_values"] is not None))
            batch["pixel_values"] = torch.stack([f["pixel_values"] if f["pixel_values"] is not None else default_image for f in features])
        
        # Handle audio - only include if at least one sample in the batch has audio
        if any(f["audio_values"] is not None for f in features):
            # If a sample has no audio, we create a tensor of zeros
            default_audio = torch.zeros_like(next(f["audio_values"] for f in features if f["audio_values"] is not None))
            batch["audio_values"] = torch.stack([f["audio_values"] if f["audio_values"] is not None else default_audio for f in features])

        # For causal language modeling, the labels are the input_ids
        batch["labels"] = batch["input_ids"].clone()
        
        return batch

class DataPipeline:
    """Multi-source data loading and preprocessing for Any-to-Text."""

    def __init__(self, model_config, tokenizer_name="meta-llama/Llama-2-7b-hf"):
        self.model_config = model_config
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize processors for each modality
        self.image_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.audio_processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

    def prepare_dataset(self, dataset_name: str, data_dir: str = "./data/cached"):
        """
        Loads, processes, and wraps a dataset for multi-modal training.
        
        Args:
            dataset_name: The friendly name of the dataset (e.g., 'coco', 'common_voice').
            data_dir: The directory where datasets were cached by `prepare_datasets.py`.
        
        Returns:
            A `MultiModalDataset` instance ready for training.
        """
        dataset_path = os.path.join(data_dir, dataset_name)
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Dataset not found at {dataset_path}. "
                f"Please run `python scripts/prepare_datasets.py {dataset_name}` first."
            )
            
        logger.info(f"Loading dataset from disk: {dataset_path}")
        hf_dataset = load_from_disk(dataset_path)
        
        # Determine column names for each modality
        column_names = hf_dataset.column_names
        text_col = "text"
        if "caption" in column_names:
            text_col = "caption"
        elif "sentence" in column_names:
            text_col = "sentence"

        image_col = "image" if "image" in column_names else None
        audio_col = "audio" if "audio" in column_names else None

        return MultiModalDataset(
            hf_dataset,
            self.tokenizer,
            self.image_processor,
            self.audio_processor,
            text_col=text_col,
            image_col=image_col,
            audio_col=audio_col
        )

    def get_data_collator(self) -> DataCollatorForMultiModal:
        """Returns the custom data collator for batching."""
        return DataCollatorForMultiModal(self.tokenizer)