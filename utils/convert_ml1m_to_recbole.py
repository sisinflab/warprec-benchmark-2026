"""
Convert MovieLens-1M ratings.csv to RecBole format.

This script converts the CSV file into three RecBole format files:
- .inter: interaction data (user_id, item_id, rating, timestamp)
- .user: user features (user_id)
- .item: item features (item_id)
"""

import pandas as pd
import os
from pathlib import Path

def convert_ml1m_to_recbole(input_path, output_dir):
    """
    Convert MovieLens-1M ratings.csv to RecBole format.
    
    Args:
        input_path: Path to ratings.csv file
        output_dir: Directory where to save RecBole format files
    """
    print(f"Reading {input_path}...")
    
    # Read the ratings CSV file
    df = pd.read_csv(input_path)
    
    print(f"Loaded {len(df)} ratings")
    print(f"Columns: {list(df.columns)}")
    print(f"Data preview:\n{df.head()}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output file names
    dataset_name = "movielens-1m"
    inter_file = os.path.join(output_dir, f"{dataset_name}.inter")
    user_file = os.path.join(output_dir, f"{dataset_name}.user")
    item_file = os.path.join(output_dir, f"{dataset_name}.item")
    
    # 1. Create .inter file (interaction data)
    # Add RecBole type annotations to column names
    print(f"\nCreating {inter_file}...")
    inter_df = df.copy()
    inter_df.columns = ['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float']
    inter_df.to_csv(inter_file, sep=',', index=False)
    print(f"✓ Saved {len(inter_df)} interactions")
    
    # 2. Create .user file (unique users)
    print(f"\nCreating {user_file}...")
    unique_users = df[['user_id']].drop_duplicates().sort_values('user_id')
    unique_users.columns = ['user_id:token']
    unique_users.to_csv(user_file, sep=',', index=False)
    print(f"✓ Saved {len(unique_users)} unique users")
    
    # 3. Create .item file (unique items)
    print(f"\nCreating {item_file}...")
    unique_items = df[['item_id']].drop_duplicates().sort_values('item_id')
    unique_items.columns = ['item_id:token']
    unique_items.to_csv(item_file, sep=',', index=False)
    print(f"✓ Saved {len(unique_items)} unique items")
    
    print("\n" + "="*50)
    print("Conversion completed successfully!")
    print("="*50)
    print(f"\nGenerated files:")
    print(f"  - {inter_file}")
    print(f"  - {user_file}")
    print(f"  - {item_file}")
    
    # Print statistics
    print(f"\nDataset statistics:")
    print(f"  - Users: {len(unique_users)}")
    print(f"  - Items: {len(unique_items)}")
    print(f"  - Interactions: {len(df)}")
    print(f"  - Sparsity: {1 - (len(df) / (len(unique_users) * len(unique_items))):.4f}")
    print(f"  - Rating range: [{df['rating'].min()}, {df['rating'].max()}]")

if __name__ == "__main__":
    # Define paths
    base_dir = Path("warprec-benchmark-2026")
    input_csv = base_dir / "dataset/movielens-1m/ratings.csv"
    output_dir = base_dir / "dataset/movielens-1m"
    
    # Convert the dataset
    convert_ml1m_to_recbole(str(input_csv), str(output_dir))
