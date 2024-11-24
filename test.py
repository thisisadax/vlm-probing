import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def create_test_image():
    """Create a simple test image"""
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    return img

def visualize_token_masks(tokens, mask):
    """Visualize which tokens correspond to image vs text"""
    plt.figure(figsize=(15, 5))
    plt.imshow([mask.numpy()], aspect='auto', cmap='binary')
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('token_mask_viz.png')
    plt.close()

def main():
    # Initialize model and processor
    processor = AutoProcessor.from_pretrained('Qwen/Qwen2-VL-7B-Instruct')
    
    # Create test inputs
    test_image = create_test_image()
    test_text = "What's in this image?"
    
    # Create message format similar to the one used in models/qwen.py
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": test_image},
                {"type": "text", "text": test_text},
            ],
        }
    ]

    # Process inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=text,
        images=test_image,
        return_tensors="pt"
    )

    # Get input IDs and decode to tokens
    input_ids = inputs.input_ids[0]
    tokens = processor.tokenizer.convert_ids_to_tokens(input_ids)

    # Print special tokens map
    print("Special tokens map:")
    print(processor.tokenizer.special_tokens_map)
    
    # Create image token mask
    image_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    
    # Find image tokens
    # Note: You might need to adjust these tokens based on actual Qwen2-VL implementation
    image_start_token = processor.tokenizer.convert_tokens_to_ids(['<image>'])[0]
    image_end_token = processor.tokenizer.convert_tokens_to_ids(['</image>'])[0]
    
    # Find positions of image tokens
    start_indices = (input_ids == image_start_token).nonzero()
    end_indices = (input_ids == image_end_token).nonzero()
    
    # Set mask for image tokens
    for start, end in zip(start_indices, end_indices):
        image_mask[start:end+1] = True
    
    # Print tokens and their mask values
    print("\nTokens and their masks:")
    for token, is_image in zip(tokens, image_mask):
        print(f"Token: {token:20} Is Image: {is_image}")
    
    # Visualize the results
    visualize_token_masks(tokens, image_mask)

if __name__ == "__main__":
    main()
