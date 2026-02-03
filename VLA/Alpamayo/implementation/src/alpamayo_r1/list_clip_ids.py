#!/usr/bin/env python3
# Script to list available clip IDs from the clip_ids.parquet file

import pandas as pd
from pathlib import Path

# Path to the parquet file
parquet_path = Path(__file__).parent.parent.parent / "notebooks" / "clip_ids.parquet"

if not parquet_path.exists():
    print(f"Error: clip_ids.parquet not found at {parquet_path}")
    print("\nTrying to download from dataset...")
    try:
        import physical_ai_av
        avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
        print("Dataset interface methods:")
        methods = [m for m in dir(avdi) if not m.startswith('_') and 'clip' in m.lower()]
        for method in methods:
            print(f"  - {method}")
    except Exception as e:
        print(f"Error: {e}")
else:
    # Read the parquet file
    print(f"Reading clip IDs from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    if 'clip_id' in df.columns:
        clip_ids = df['clip_id'].tolist()
        
        print(f"\nFound {len(clip_ids)} clips in the dataset:\n")
        print("=" * 80)
        
        # Print first 50 clip IDs with their indices
        for i, clip_id in enumerate(clip_ids[:50]):
            print(f"{i:3d}. {clip_id}")
        
        if len(clip_ids) > 50:
            print(f"\n... and {len(clip_ids) - 50} more clips")
        
        print(f"\nTotal clips: {len(clip_ids)}")
        print("=" * 80)
        
        print("\nTo use a clip ID, update test_inference.py:")
        print('  clip_id = "your-chosen-clip-id"')
        print("\nExamples:")
        print(f'  clip_id = "{clip_ids[0]}"  # First clip (index 0)')
        if len(clip_ids) > 1:
            print(f'  clip_id = "{clip_ids[1]}"  # Second clip (index 1)')
        if len(clip_ids) > 774:
            print(f'  clip_id = "{clip_ids[774]}"  # Clip used in notebook (index 774)')
    else:
        print(f"Error: 'clip_id' column not found in parquet file")
        print(f"Available columns: {df.columns.tolist()}")


