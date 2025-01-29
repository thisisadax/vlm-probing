import os
from functools import reduce
import traceback
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from models.model import Model
from tasks.task_utils import Task
from utils import get_attr_or_item

class InternVL(Model):
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
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.activations = {}
        self.save_counter = 1
        self.save_interval = save_interval
        self.prompt = Path
        self.model_name = model_name

        # Initialize model and tokenizer
        model_path = "OpenGVLab/InternVL2_5-8B"
        self.model = AutoModel.from_pretrained(model_path,
                                               torch_dtype=torch.bfloat16,
                                               low_cpu_mem_usage=True,
                                               use_flash_attn=True,
                                               trust_remote_code=True,
                                               device_map='auto'
                                               ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                       trust_remote_code=True,
                                                       use_fast=False)

        # Set probe layers - either use provided config or detect all MLPs
        self.probe_layers = probe_layers if probe_layers is not None else self._detect_all_encoder_layers()
        
        # Register hooks if probe_layers are specified
        if self.probe_layers:
            self._register_hooks()

    def _detect_all_encoder_layers(self) -> Dict:
        '''Detect all encoder layers in the model and create a probe configuration.'''
        probe_config = {}
        
        # Get number of layers in the model
        num_layers = len(self.model.vision_model.encoder.layers)
        
        # Create probe config for each layer
        for layer_idx in range(num_layers):
            layer_name = f'layer-{layer_idx}'
            probe_config[layer_name] = ['vision_model', 'encoder', 'layers', str(layer_idx)]
            
        return probe_config
    

    def _get_activations(self, name):
        def hook(model, input, output):
            try:
                if isinstance(output, tuple):
                    output = output[0]
                # Only collect if sequence length > 1 (not generation phase)
                if output.size(1) > 1:
                    output = output.detach().cpu()
                    try:
                        self.activations[name].append(output)
                    except KeyError:
                        self.activations[name] = [output]     
            except Exception as e:
                print(f'Error in hook for {name}: {str(e)}')
        return hook


    def _register_hooks(self):
        '''Register forward hooks for specified layers.'''
        for layer_name, layer_path in self.probe_layers.items():
            if isinstance(layer_path, str):
                layer_path = layer_path.split('.')
            target = reduce(get_attr_or_item, layer_path, self.model)
            target.register_forward_hook(self._get_activations(layer_name))

    def run_batch(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        '''Run inference on a batch of DataFrame rows.'''
        # Convert rows to messages
        image_paths = batch_df.path.values
        generation_config = dict(max_new_tokens=1024, do_sample=True)

        # batch inference, single image per sample
        pixel_values = [load_image(image_paths[i], max_num=1).to(torch.bfloat16).cuda()
                        for i in range(len(image_paths))]
        num_patches_list = [pixel_values[i].size(0) for i in range(len(pixel_values))]
        pixel_values = torch.cat(pixel_values, dim=0)

        questions = [self.task.get_prompt(row) for _, row in batch_df.iterrows()]
        responses = self.model.batch_chat(self.tokenizer, pixel_values,
                                    num_patches_list=num_patches_list,
                                    questions=questions,
                                    generation_config=generation_config)
        
        generated_texts = [response for response in responses]

        #TODO: Compute image masks for the batch

        print(f'Number of objects: {batch_df.n_objects.values[0]}')
        print(f'Model estimate: {generated_texts[0]}')
        print(f'path: {batch_df.path.values[0].split("/")[-1]}')
        # print(f'path: {batch_df.object.values}')
        print('\n')
        
        # Add responses to DataFrame
        batch_df['response'] = generated_texts
        return batch_df

    def run(self) -> pd.DataFrame:
        '''Run inference on all trials in batches and return updated DataFrame.'''
        # Split dataframe into batches
        batches = np.array_split(
            self.task.results_df,
            np.ceil(len(self.task.results_df)/self.batch_size)
        )
        
        # Process each batch
        processed_batches = []
        #for i, batch_df in tqdm(enumerate(batches), total=len(batches)):
        for i, batch_df in enumerate(batches):
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
                self.task.model_name,
                'image_mask.pt'
            )
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            torch.save(mask_tensor, mask_path)
        
        return final_df

    def save_activations(self):
        '''Save extracted activations to disk as PyTorch tensors and clear buffer.'''
        outpath = os.path.join(
            self.task.output_dir, 
            self.task.task_name,
            self.task.model_name,
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


# image util functions

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values