import os
from functools import reduce
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Union
import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from models.model import Model
from tasks.task import Task
from utils import get_attr_or_item


class Qwen(Model):
    def __init__(
        self,
        task: Task,
        max_tokens: int = 128,
        batch_size: int = 32,
        probe_layers: Dict = None,
        device: str = None,
        model_name: str = None,
        save_interval: int = 10
    ):
        self.image_masks = []
        super().__init__(task)
        self.max_new_tokens = max_tokens
        self.batch_size = batch_size
        self.probe_layers = probe_layers
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.activations = {}
        self.save_counter = 1
        self.save_interval = save_interval
        self.prompt = Path
        self.model_name = model_name
        
        # Initialize model and processor
        self.model = Qwen2VLForConditionalGeneration.from_pretrained('Qwen/Qwen2-VL-7B-Instruct', 
                                                                     torch_dtype='auto', 
                                                                     device_map='auto', 
                                                                     local_files_only=True)
        self.processor = AutoProcessor.from_pretrained('Qwen/Qwen2-VL-7B-Instruct')
        self.model.to(self.device)
        
        # Register hooks if probe_layers are specified
        if self.probe_layers:
            self.register_hooks()
    

    def getActivations(self, name):
        def hook(model, input, output):
            try:
                if isinstance(output, tuple):
                    output = output[0]
                
                # Only collect if sequence length > 1 (not generation phase)
                if output.size(1) > 1:
                    try:
                        self.activations[name].append(output.detach().cpu())
                    except KeyError:
                        self.activations[name] = [output.detach().cpu()]
                    #print(f'{name} Shape: {output.shape}')
                    
            except Exception as e:
                print(f'Error in hook for {name}: {str(e)}')
                
        return hook


    def register_hooks(self):
        """Register forward hooks for specified layers."""
        for layer_name, layer_path in self.probe_layers.items():
            if isinstance(layer_path, str):
                layer_path = layer_path.split('.')
            target = reduce(get_attr_or_item, layer_path, self.model)
            target.register_forward_hook(self.getActivations(layer_name))


    def get_image_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute binary mask indicating image token positions."""
        image_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        
        # Get vision special token IDs
        image_start_token = self.processor.tokenizer.convert_tokens_to_ids(['<|vision_start|>'])[0]
        image_end_token = self.processor.tokenizer.convert_tokens_to_ids(['<|vision_end|>'])[0]
        
        # Find positions of image tokens
        start_indices = torch.where(input_ids == image_start_token)[0]
        end_indices = torch.where(input_ids == image_end_token)[0]
        
        # Set mask for image tokens
        for start, end in zip(start_indices, end_indices):
            image_mask[start:end+1] = True
            
        return image_mask

    def run_batch(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        """Run inference on a batch of DataFrame rows."""
        # Convert rows to messages
        batch_messages = [self.encode_trial(row) for _, row in batch_df.iterrows()]
        
        # Prepare inputs
        texts = [
            self.processor.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=True
            ) for msg in batch_messages
        ]
        
        image_inputs, video_inputs = process_vision_info(batch_messages)
        
        # Process all inputs
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors='pt'
        ).to(self.device)

        # Compute image masks for the batch
        batch_masks = [self.get_image_mask(ids) for ids in inputs.input_ids]
        self.image_masks.extend(batch_masks)

        # Generate outputs
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens
            )
        
        # Trim input tokens from output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # Decode outputs
        outputs = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        # Add responses to DataFrame
        batch_df['response'] = outputs
        return batch_df


    def encode_trial(self, row) -> List[dict]:
        """Convert a dataframe row into the expected message format."""
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{row['path']}"},
                    {"type": "text", "text": self.task.prompt},
                ],
            }
        ]

    def run(self) -> pd.DataFrame:
        """Run inference on all trials in batches and return updated DataFrame."""
        # Split dataframe into batches
        batches = np.array_split(
            self.task.results_df,
            np.ceil(len(self.task.results_df)/self.batch_size)
        )
        
        # Process each batch
        processed_batches = []
        for i, batch_df in tqdm(enumerate(batches), total=len(batches)):
            # Run inference on batch
            processed_batch = self.run_batch(batch_df)
            processed_batches.append(processed_batch)
            
            # Save activations based on save_interval
            if self.probe_layers and (i+1) % self.save_interval == 0:
                self.save_activations()
                self.save_counter += 1
        
        # Combine all processed batches
        final_df = pd.concat(processed_batches, axis=0, ignore_index=True)
        
        # Save results to CSV
        final_df.to_csv(self.task.results_path, index=False)
        
        # Save concatenated image masks
        if self.image_masks:
            mask_tensor = torch.cat(self.image_masks, dim=0)
            mask_path = os.path.join(
                self.task.output_dir,
                self.task.task_name,
                Path(self.task.model_name),
                'image_mask.pt'
            )
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            torch.save(mask_tensor, mask_path)
        
        return final_df

    def save_activations(self):
        """Save extracted activations to disk as PyTorch tensors and clear buffer."""
        outpath = os.path.join(
            self.task.output_dir, 
            self.task.task_name,
            Path(self.task.model_name),
            'activations'
        )
        os.makedirs(outpath, exist_ok=True)
        
        for layer, layer_activations in self.activations.items():
            try:
                # Stack activations maintaining BxTxD shape
                activations = torch.cat(layer_activations, dim=0)
                
                # Create layer-specific directory
                layer_dir = os.path.join(outpath, layer)
                os.makedirs(layer_dir, exist_ok=True)
                
                # Save to disk as PyTorch tensor with unique counter
                save_path = os.path.join(layer_dir, f'{self.save_counter:04d}.pt')
                torch.save(activations, save_path)
                
            except Exception as e:
                print(f'Error saving {layer}: {str(e)}')
                traceback.print_exc()
        
        # Clear the activation buffer after saving
        self.activations = {}
