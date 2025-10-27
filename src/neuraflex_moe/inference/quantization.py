"""Model quantization utilities"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from typing import Optional


class ModelQuantizer:
    """Quantize models for efficient deployment"""
    
    @staticmethod
    def quantize_4bit(model_path: str, output_path: str):
        """Quantize model to 4-bit"""
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            device_map="auto"
        )
        
        model.save_pretrained(output_path, safe_serialization=True)
        return model
    
    @staticmethod
    def quantize_8bit(model_path: str, output_path: str):
        """Quantize model to 8-bit"""
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=True,
            device_map="auto"
        )
        
        model.save_pretrained(output_path, safe_serialization=True)
        return model
    
    @staticmethod
    def export_onnx(model: nn.Module, output_path: str, dummy_input: torch.Tensor):
        """Export model to ONNX format"""
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input_ids'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'logits': {0: 'batch_size', 1: 'sequence'}
            }
        )
