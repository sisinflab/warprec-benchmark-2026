import pandas as pd
import sys

def convert_csv_to_tsv(input_csv, output_tsv):
    """
    Legge un file CSV, rimuove l'header e lo salva come TSV.
    Le colonne vengono ordinate come: user_id, item_id, rating, timestamp
    """
    # Leggi il CSV
    df = pd.read_csv(input_csv)
    
    # Identifica le colonne necessarie
    # Cerca nomi comuni per le colonne
    column_mapping = {}
    
    for col in df.columns:
        col_lower = col.lower()
        if 'user' in col_lower and 'id' in col_lower:
            column_mapping['user_id'] = col
        elif 'item' in col_lower and 'id' in col_lower:
            column_mapping['item_id'] = col
        elif 'rating' in col_lower or 'score' in col_lower:
            column_mapping['rating'] = col
        elif 'time' in col_lower or 'timestamp' in col_lower:
            column_mapping['timestamp'] = col
    
    # Se non tutte le colonne sono state trovate, prova con i nomi diretti
    if len(column_mapping) < 4:
        column_mapping = {
            'user_id': df.columns[0],
            'item_id': df.columns[1],
            'rating': df.columns[2],
            'timestamp': df.columns[3] if len(df.columns) > 3 else None
        }
    
    # Rinomina le colonne e seleziona l'ordine corretto
    df_renamed = pd.DataFrame()
    df_renamed['user_id'] = df[column_mapping['user_id']]
    df_renamed['item_id'] = df[column_mapping['item_id']]
    df_renamed['rating'] = df[column_mapping['rating']]
    if column_mapping.get('timestamp'):
        df_renamed['timestamp'] = df[column_mapping['timestamp']]
    
    # Salva come TSV senza header
    df_renamed.to_csv(output_tsv, sep='\t', header=False, index=False)
    print(f"Convertito {input_csv} in {output_tsv}")
    print(f"Righe: {len(df_renamed)}, Colonne: {len(df_renamed.columns)}")

if __name__ == "__main__":
    
    input_file = "../dataset/netflix-prize-100m/ratings_processed.csv"
    output_file = "../dataset/netflix-prize-100m/ratings.tsv"
    
    convert_csv_to_tsv(input_file, output_file)
