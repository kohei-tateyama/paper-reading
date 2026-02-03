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

# End-to-end example script for the inference pipeline:
# This script loads a dataset, runs inference, and computes the minADE.
# It can be used to test the inference pipeline.

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


# Load available clip IDs and select a random one
clip_ids_path = Path(__file__).parent.parent.parent / "notebooks" / "clip_ids.parquet"
clip_ids = pd.read_parquet(clip_ids_path)["clip_id"].tolist()
clip_id = random.choice(clip_ids)

print(f"Randomly selected clip_id: {clip_id} (from {len(clip_ids)} available clips)")
print(f"Loading dataset...")
data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
print("Dataset loaded.")

# Save input images
output_dir = Path("output_images")
output_dir.mkdir(exist_ok=True)

image_frames = data["image_frames"]  # Shape: (N_cameras, num_frames, 3, H, W)
print(f"Original image frames shape: {image_frames.shape}")
print(f"  -> {image_frames.shape[0]} cameras, {image_frames.shape[1]} frames per camera")


# Save flattened images (what the model actually sees)
flattened_frames = data["image_frames"].flatten(0, 1)
print(f"Flattened image frames shape (model input): {flattened_frames.shape}")

flattened_dir = output_dir / "model_input"
flattened_dir.mkdir(exist_ok=True)

for idx in range(flattened_frames.shape[0]):
    img_tensor = flattened_frames[idx]
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    
    # Convert from float [0, 1] to uint8 [0, 255] if needed
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    else:
        img_np = img_np.astype(np.uint8)
    
    # Save image with sequential numbering
    img = Image.fromarray(img_np)
    img.save(flattened_dir / f"input_{idx:02d}.png")

print(f"Saved {flattened_frames.shape[0]} flattened images to {flattened_dir}/ (these are what the model sees)")

messages = helper.create_message(data["image_frames"].flatten(0, 1))

# Store flattened images as PIL Images for later use
pil_images = []
for idx in range(flattened_frames.shape[0]):
    img_tensor = flattened_frames[idx]
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    else:
        img_np = img_np.astype(np.uint8)
    
    pil_images.append(Image.fromarray(img_np))

model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to("cuda")
processor = helper.get_processor(model.tokenizer)

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

model_inputs = helper.to_device(model_inputs, "cuda")

torch.cuda.manual_seed_all(42)
with torch.autocast("cuda", dtype=torch.bfloat16):
    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
        data=model_inputs,
        top_p=0.98,
        temperature=0.6,
        num_traj_samples=1,  # Feel free to raise this for more output trajectories and CoC traces.
        max_generation_length=256,
        return_extra=True,
    )

# the size is [batch_size, num_traj_sets, num_traj_samples]
print("Chain-of-Causation (per trajectory):\n", extra["cot"][0])

gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
min_ade = diff.min()
print("minADE:", min_ade, "meters")
print(
    "Note: VLA-reasoning models produce nondeterministic outputs due to trajectory sampling, "
    "hardware differences, etc. With num_traj_samples=1 (set for GPU memory compatibility), "
    "variance in minADE is expected. For visual sanity checks, see notebooks/inference.ipynb"
)

# Create a patched image with all inputs and model output text
num_images = len(pil_images)
if num_images > 0:
    # Get dimensions of first image
    img_width, img_height = pil_images[0].size
    
    # Fixed 4x4 grid layout
    cols = 4
    rows = 4
    
    # Create margin for text at the top
    margin_height = 1000
    text_padding = 20
    
    # Create the combined image
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
    
    # Add text on top margin
    draw = ImageDraw.Draw(combined_image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 70)
    except:
        font = ImageFont.load_default()
    
    # Format the model output text
    cot_output = extra["cot"][0]
    text_lines = ["Model Output (Chain-of-Causation):"]
    for i, cot in enumerate(cot_output):
        # Convert to string if it's a numpy array, then replace newlines
        cot_str = str(cot) if not isinstance(cot, str) else cot
        cot_flat = cot_str.replace('\n', ' ').replace('\r', ' ')
        text_lines.append(f"{i+1}. {cot_flat}")
    
    # Draw text
    y_offset = text_padding
    for line in text_lines:
        # Wrap long lines
        max_width = combined_width - 2 * text_padding
        if draw.textlength(line, font=font) > max_width:
            # Simple word wrapping
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
    
    # Save the combined image with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_path = output_dir / f"output_{timestamp}.png"
    combined_image.save(combined_path)
    print(f"\nSaved combined visualization to {combined_path}")
    print(f"Image shows {num_images} input frames in a {rows}x{cols} grid with model output on top")
