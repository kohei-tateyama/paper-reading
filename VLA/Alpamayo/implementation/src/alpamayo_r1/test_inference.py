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

"""Example script showing how to use the modular inference functions."""

import pandas as pd
import random
from pathlib import Path

from alpamayo_r1.inference import load_model, run_inference, save_visualization


# Load available clip IDs
clip_ids_path = Path(__file__).parent.parent.parent / "notebooks" / "clip_ids.parquet"
clip_ids = pd.read_parquet(clip_ids_path)["clip_id"].tolist()

# Load model once (this is the expensive operation)
print("Loading model...")
model, processor = load_model()
print("Model loaded successfully!")

# Run inference on a random clip
clip_id = random.choice(clip_ids)
print(f"\nRunning inference on clip: {clip_id} (from {len(clip_ids)} available clips)")

result = run_inference(model, processor, clip_id, num_traj_samples=1)

print(f"\nChain-of-Causation (per trajectory):\n", result["cot"])
print(f"minADE: {result['min_ade']:.4f} meters")

# Save visualization
viz_path = save_visualization(result)
print(f"\nSaved combined visualization to {viz_path}")

print(
    "\nNote: VLA-reasoning models produce nondeterministic outputs due to trajectory sampling, "
    "hardware differences, etc. With num_traj_samples=1 (set for GPU memory compatibility), "
    "variance in minADE is expected. For visual sanity checks, see notebooks/inference.ipynb"
)
