# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Modular inference functions for Alpamayo-R1."""

import torch
import numpy as np
import pandas as pd
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper


def load_model(device="cuda", dtype=torch.bfloat16):
    """Load the Alpamayo-R1 model and processor.
    
    Args:
        device: Device to load model on (default: "cuda")
        dtype: Data type for model (default: torch.bfloat16)
    
    Returns:
        tuple: (model, processor)
    """
    model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=dtype).to(device)
    processor = helper.get_processor(model.tokenizer)
    return model, processor


def run_inference(
    model,
    processor,
    clip_id,
    t0_us=5_100_000,
    num_traj_samples=1,
    top_p=0.98,
    temperature=0.6,
    max_generation_length=256,
    device="cuda",
    seed=42,
):
    """Run inference on a single clip.
    
    Args:
        model: Pre-loaded Alpamayo-R1 model
        processor: Pre-loaded processor
        clip_id: Clip ID to run inference on
        t0_us: Timestamp in microseconds (default: 5_100_000)
        num_traj_samples: Number of trajectory samples to generate (default: 1)
        top_p: Top-p sampling parameter (default: 0.98)
        temperature: Sampling temperature (default: 0.6)
        max_generation_length: Maximum generation length (default: 256)
        device: Device to run inference on (default: "cuda")
        seed: Random seed (default: 42)
    
    Returns:
        dict: Dictionary containing:
            - pred_xyz: Predicted trajectory positions
            - pred_rot: Predicted trajectory rotations
            - cot: Chain-of-Causation reasoning traces
            - min_ade: Minimum Average Displacement Error
            - data: Raw input data
            - clip_id: The clip ID used
    """
    # Load data
    data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
    
    # Prepare messages
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    
    # Process inputs
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    
    model_inputs = helper.to_device(model_inputs, device)
    
    # Run inference
    torch.cuda.manual_seed_all(seed)
    with torch.autocast(device, dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=top_p,
            temperature=temperature,
            num_traj_samples=num_traj_samples,
            max_generation_length=max_generation_length,
            return_extra=True,
        )
    
    # Compute minADE
    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    min_ade = diff.min()
    
    return {
        "pred_xyz": pred_xyz,
        "pred_rot": pred_rot,
        "cot": extra["cot"][0],
        "min_ade": min_ade,
        "data": data,
        "clip_id": clip_id,
    }


def save_visualization(
    result,
    output_dir="output_images",
    font_size=100,
    margin_height=1000,
):
    """Save a visualization of the input images and model output.
    
    Args:
        result: Result dictionary from run_inference()
        output_dir: Output directory for images (default: "output_images")
        font_size: Font size for text (default: 100)
        margin_height: Height of margin for text (default: 1000)
    
    Returns:
        Path: Path to the saved visualization
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Convert image frames to PIL images
    flattened_frames = result["data"]["image_frames"].flatten(0, 1)
    pil_images = []
    
    for idx in range(flattened_frames.shape[0]):
        img_tensor = flattened_frames[idx]
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
        
        pil_images.append(Image.fromarray(img_np))
    
    if len(pil_images) == 0:
        return None
    
    # Create combined image
    img_width, img_height = pil_images[0].size
    cols = 4
    rows = 4
    text_padding = 20
    
    combined_width = cols * img_width
    combined_height = margin_height + rows * img_height
    combined_image = Image.new('RGB', (combined_width, combined_height), color='white')
    
    # Paste images into grid
    for idx, img in enumerate(pil_images):
        row = idx // cols
        col = idx % cols
        x = col * img_width
        y = margin_height + row * img_height
        combined_image.paste(img, (x, y))
    
    # Add text
    draw = ImageDraw.Draw(combined_image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Format output text
    cot_output = result["cot"]
    text_lines = [
        f"Clip ID: {result['clip_id']}",
        f"minADE: {result['min_ade']:.4f} meters",
        "Model Output (Chain-of-Causation):"
    ]
    for i, cot in enumerate(cot_output):
        cot_str = str(cot) if not isinstance(cot, str) else cot
        cot_flat = cot_str.replace('\n', ' ').replace('\r', ' ')
        text_lines.append(f"{i+1}. {cot_flat}")
    
    # Draw text with wrapping
    y_offset = text_padding
    for line in text_lines:
        max_width = combined_width - 2 * text_padding
        if draw.textlength(line, font=font) > max_width:
            words = line.split()
            current_line = words[0] if words else ""
            for word in words[1:]:
                test_line = current_line + " " + word
                if draw.textlength(test_line, font=font) <= max_width:
                    current_line = test_line
                else:
                    draw.text((text_padding, y_offset), current_line, fill='black', font=font)
                    y_offset += 90
                    current_line = "  " + word
            draw.text((text_padding, y_offset), current_line, fill='black', font=font)
        else:
            draw.text((text_padding, y_offset), line, fill='black', font=font)
        y_offset += 90
    
    # Save image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_path = output_dir / f"output_{timestamp}.png"
    combined_image.save(combined_path)
    
    return combined_path


def main():
    # Load available clip IDs
    clip_ids_path = Path(__file__).parent.parent.parent / "notebooks" / "clip_ids.parquet"
    clip_ids = pd.read_parquet(clip_ids_path)["clip_id"].tolist()
    
    # Load model ONCE - this is the expensive operation
    print("Loading model...")
    model, processor = load_model()
    print("Model loaded successfully!\n")
    
    # Run inference on multiple clips
    num_clips = 50  # Change this to run on more/fewer clips
    selected_clips = random.sample(clip_ids, num_clips)
    
    results = []
    for i, clip_id in enumerate(selected_clips, 1):
        print(f"[{i}/{num_clips}] Running inference on clip: {clip_id}")
        
        # Run inference - model is already loaded, so this is fast
        result = run_inference(
            model,
            processor,
            clip_id,
            num_traj_samples=1,
            t0_us=5_100_000
        )
        
        print(f"  minADE: {result['min_ade']:.4f} meters")
        print(f"  CoT: {result['cot'][0][:100]}...")  # Print first 100 chars
        
        # Save visualization
        viz_path = save_visualization(result)
        print(f"  Saved to: {viz_path}\n")
        
        results.append(result)
    
    # Summary statistics
    print("=" * 80)
    print("Summary Statistics:")
    print(f"Total clips processed: {len(results)}")
    ade_values = [r['min_ade'] for r in results]
    print(f"Average minADE: {sum(ade_values) / len(ade_values):.4f} meters")
    print(f"Best minADE: {min(ade_values):.4f} meters")
    print(f"Worst minADE: {max(ade_values):.4f} meters")
    print("=" * 80)


if __name__ == "__main__":
    main()