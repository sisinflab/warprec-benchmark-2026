import pandas as pd
import glob
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm


def process_file(filepath):
    """
    Process a single Netflix training set file.
    
    Args:
        filepath: Path to the .txt file
        
    Returns:
        pandas DataFrame with columns: item_id, user_id, rating, date
    """
    with open(filepath, 'r') as f:
        # First line contains the movie ID (item_id)
        first_line = f.readline().strip()
        item_id = first_line.rstrip(':')
        
        # Read remaining lines as CSV
        data = []
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(',')
                if len(parts) == 3:
                    data.append([item_id] + parts)
    
    if data:
        df = pd.DataFrame(data, columns=['item_id', 'user_id', 'rating', 'date'])
        return df
    return None


def merge_netflix_files(input_dir, output_file, chunk_size=500):
    """
    Merge all Netflix training set files into a single shuffled CSV.
    
    Args:
        input_dir: Directory containing the training_set/*.txt files
        output_file: Output CSV file path
        chunk_size: Number of files to process before writing to disk
    """
    print(f"Searching for files in: {input_dir}")
    
    # Get all .txt files
    pattern = os.path.join(input_dir, "training_set", "*.txt")
    files = sorted(glob.glob(pattern))
    
    print(f"Found {len(files)} files to process")
    
    if len(files) == 0:
        print("Error: No files found!")
        return
    
    # Process files in chunks to manage memory
    all_chunks = []
    temp_files = []
    
    print("Processing files...")
    for i in tqdm(range(0, len(files), chunk_size), desc="Processing chunks"):
        chunk_files = files[i:i + chunk_size]
        dfs = []
        
        for filepath in chunk_files:
            df = process_file(filepath)
            if df is not None:
                dfs.append(df)
        
        if dfs:
            # Concatenate this chunk
            chunk_df = pd.concat(dfs, ignore_index=True)
            
            # Save to temporary file
            temp_file = f"{output_file}.temp_{i//chunk_size}.csv"
            chunk_df.to_csv(temp_file, index=False)
            temp_files.append(temp_file)
            
            print(f"  Chunk {i//chunk_size + 1}: {len(chunk_df):,} ratings saved to temp file")
            
            # Clear memory
            del chunk_df
            del dfs
    
    # Now merge all temporary files and shuffle
    print("\nMerging temporary files and shuffling...")
    
    # Read all temp files, shuffle, and write final output
    first_chunk = True
    for temp_file in tqdm(temp_files, desc="Final merge"):
        chunk_df = pd.read_csv(temp_file)
        
        # Shuffle this chunk
        chunk_df = chunk_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Write to final file
        if first_chunk:
            chunk_df.to_csv(output_file, index=False, mode='w')
            first_chunk = False
        else:
            chunk_df.to_csv(output_file, index=False, mode='a', header=False)
        
        # Remove temp file
        os.remove(temp_file)
        del chunk_df
    
    print(f"\n✓ Successfully merged all files into: {output_file}")
    
    # Display summary
    print("\nReading final file to get statistics...")
    total_lines = sum(1 for _ in open(output_file)) - 1  # Subtract header
    print(f"Total ratings: {total_lines:,}")
    
    # Show first few rows
    print("\nFirst 10 rows of the shuffled dataset:")
    df_sample = pd.read_csv(output_file, nrows=10)
    print(df_sample)
    print(f"\nData types:\n{df_sample.dtypes}")


if __name__ == "__main__":
    # Configuration
    BASE_DIR = "warprec-benchmark-2026/dataset/netflix-prize-100m"
    INPUT_DIR = os.path.join(BASE_DIR, "download")
    OUTPUT_FILE = os.path.join(BASE_DIR, "netflix_training_set_shuffled.csv")
    
    print("=" * 70)
    print("Netflix Prize Training Set Merger")
    print("=" * 70)
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output file: {OUTPUT_FILE}")
    print("=" * 70)
    print()
    
    # Run the merge
    merge_netflix_files(INPUT_DIR, OUTPUT_FILE, chunk_size=500)
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
