import pandas as pd
from datetime import datetime

# Leggi il file CSV
input_file = 'warprec-benchmark-2026/dataset/netflix-prize-100m/ratings.csv'
output_file = 'warprec-benchmark-2026/dataset/netflix-prize-100m/ratings_processed.csv'

print("Lettura del file...")
df = pd.read_csv(input_file)

print(f"Numero di righe: {len(df)}")
print(f"Colonne originali: {df.columns.tolist()}")

# Converti la colonna 'date' in timestamp (Unix timestamp)
print("Conversione date in timestamp...")
df['timestamp'] = pd.to_datetime(df['date']).astype(int) // 10**9

# Riordina le colonne nell'ordine richiesto: user_id, item_id, rating, timestamp
print("Riordinamento colonne...")
df = df[['user_id', 'item_id', 'rating', 'timestamp']]

# Salva il file processato
print(f"Salvataggio del file processato in: {output_file}")
df.to_csv(output_file, index=False)

print("Completato!")
print(f"\nPrime righe del file processato:")
print(df.head(10))
