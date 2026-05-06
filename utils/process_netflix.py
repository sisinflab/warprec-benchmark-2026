import pandas as pd
from datetime import datetime

input_file = 'warprec-benchmark-2026/dataset/netflix-prize-100m/ratings.csv'
output_file = 'warprec-benchmark-2026/dataset/netflix-prize-100m/ratings_processed.csv'

print("Reading file...")
df = pd.read_csv(input_file)

print(f"Number of rows: {len(df)}")
print(f"Original columns: {df.columns.tolist()}")

print("Converting dates to Unix timestamps...")
df['timestamp'] = pd.to_datetime(df['date']).astype(int) // 10**9

# Reorder columns to: user_id, item_id, rating, timestamp
print("Reordering columns...")
df = df[['user_id', 'item_id', 'rating', 'timestamp']]

print(f"Saving processed file to: {output_file}")
df.to_csv(output_file, index=False)

print("Done!")
print("\nFirst rows of the processed file:")
print(df.head(10))
