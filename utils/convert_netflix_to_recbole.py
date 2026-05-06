#!/usr/bin/env python3
"""
Convert Netflix Prize 100M dataset to RecBole format.

Usage:
    python convert_netflix_to_recbole.py
"""

import pandas as pd
import os
from pathlib import Path

# Paths
NETFLIX_CSV = "warprec-benchmark-2026/dataset/netflix-prize-100m/ratings_processed.csv"
OUTPUT_DIR = "warprec-benchmark-2026/dataset/netflix-prize-100m"

# Output filenames
INTER_FILE = os.path.join(OUTPUT_DIR, "netflix-prize-100m.inter")
USER_FILE = os.path.join(OUTPUT_DIR, "netflix-prize-100m.user")
ITEM_FILE = os.path.join(OUTPUT_DIR, "netflix-prize-100m.item")

# Chunk size for processing large file
CHUNK_SIZE = 1_000_000


def convert_to_recbole():
    """Convert Netflix ratings to RecBole format."""
    
    print(f"Converting {NETFLIX_CSV} to RecBole format...")
    print(f"Processing in chunks of {CHUNK_SIZE:,} rows...")
    
    # Sets to collect unique users and items
    unique_users = set()
    unique_items = set()
    
    # First pass: Write .inter file and collect unique users/items
    print("\nStep 1: Creating .inter file and collecting unique IDs...")
    
    first_chunk = True
    total_rows = 0
    
    for chunk_num, chunk in enumerate(pd.read_csv(NETFLIX_CSV, chunksize=CHUNK_SIZE), 1):
        # Convert to RecBole format with tab separator
        chunk_inter = chunk[['user_id', 'item_id', 'rating', 'timestamp']]
        
        # Collect unique users and items
        unique_users.update(chunk['user_id'].unique())
        unique_items.update(chunk['item_id'].unique())
        
        # Write .inter file
        if first_chunk:
            # Write header with type annotations
            with open(INTER_FILE, 'w') as f:
                f.write("user_id:token,item_id:token,rating:float,timestamp:float\n")
            first_chunk = False
        
        # Append data (comma-separated)
        chunk_inter.to_csv(INTER_FILE, sep=',', index=False, header=False, mode='a')
        
        total_rows += len(chunk)
        print(f"  Processed chunk {chunk_num}: {total_rows:,} rows, "
              f"{len(unique_users):,} unique users, {len(unique_items):,} unique items")
    
    print(f"\n✓ Created {INTER_FILE}")
    print(f"  Total rows: {total_rows:,}")
    
    # Step 2: Write .user file
    print(f"\nStep 2: Creating .user file...")
    sorted_users = sorted(unique_users)
    with open(USER_FILE, 'w') as f:
        f.write("user_id:token\n")
        for user_id in sorted_users:
            f.write(f"{user_id}\n")
    
    print(f"✓ Created {USER_FILE}")
    print(f"  Total unique users: {len(sorted_users):,}")
    
    # Step 3: Write .item file
    print(f"\nStep 3: Creating .item file...")
    sorted_items = sorted(unique_items)
    with open(ITEM_FILE, 'w') as f:
        f.write("item_id:token\n")
        for item_id in sorted_items:
            f.write(f"{item_id}\n")
    
    print(f"✓ Created {ITEM_FILE}")
    print(f"  Total unique items: {len(sorted_items):,}")
    
    print("\n" + "="*60)
    print("✓ Conversion complete!")
    print("="*60)
    print(f"\nOutput files:")
    print(f"  - {INTER_FILE}")
    print(f"  - {USER_FILE}")
    print(f"  - {ITEM_FILE}")


if __name__ == "__main__":
    convert_to_recbole()
