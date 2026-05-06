import pandas as pd
import numpy as np
import sys
import os

def process_csv_file(file_path):
    """Process a single CSV file and return the calculated metrics."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"Error reading '{file_path}': {e}")
        return None

    # Calculate the mean over the entire file (or per project_name if present)
    # For simplicity, we calculate the global mean
    means = df.mean(numeric_only=True)
    
    # Extract the file name for identification
    file_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
    
    metrics = {
        'name': file_name,
        'duration': means.get('duration', 0),
        'emissions': means.get('emissions', 0),
        'emissions_rate': means.get('emissions_rate', 0) * 3600,  # Convert to kg/h
        'cpu_power': means.get('cpu_power', 0),
        'gpu_power': means.get('gpu_power', 0),
        'cpu_energy': means.get('cpu_energy', 0),
        'gpu_energy': means.get('gpu_energy', 0),
        'ram_energy': means.get('ram_energy', 0),
        'energy_consumed': means.get('energy_consumed', 0),
        'water_consumed': means.get('water_consumed', 0),
        'ram_used_gb': means.get('ram_used_gb', 0)
    }
    
    return metrics

def format_value(value, decimals=4, is_gpu=False):
    """Format a value for LaTeX, handling N/A for GPU if necessary."""
    if is_gpu and (value == 0 or np.isnan(value)):
        return "N/A"
    return f"{value:.{decimals}f}"

def generate_unified_latex_table(csv_files):
    """Generate a unified LaTeX table with all provided CSV files."""
    all_metrics = []
    
    print(f"\n=== Processing {len(csv_files)} CSV files ===\n")
    
    # Process each CSV file
    for csv_file in csv_files:
        print(f"Processing: {csv_file}")
        metrics = process_csv_file(csv_file)
        if metrics:
            all_metrics.append(metrics)
            print(f"  ✓ Completed: {metrics['name']}\n")
        else:
            print(f"  ✗ Error processing\n")
    
    if not all_metrics:
        print("Error: No file was successfully processed.")
        return
    
    # Generate the table header
    num_cols = len(all_metrics)
    col_names = " & ".join([m['name'] for m in all_metrics])
    
    print("\n" + "=" * 80)
    print("UNIFIED LATEX TABLE")
    print("=" * 80 + "\n")
    
    print("% Copy and paste into your LaTeX table:")
    print(f"% Number of columns: {num_cols}\n")
    
    # Generate the table rows
    metrics_config = [
        ('Duration (s)', 'duration', 4, False),
        ('Emissions (kg CO$_2$eq)', 'emissions', 6, False),
        ('Emissions Rate (kg CO$_2$eq/h)', 'emissions_rate', 6, False),
        ('CPU Power (W)', 'cpu_power', 4, False),
        ('GPU Power (W)', 'gpu_power', 4, True),
        ('CPU Energy (kWh)', 'cpu_energy', 6, False),
        ('GPU Energy (kWh)', 'gpu_energy', 6, True),
        ('RAM Energy (kWh)', 'ram_energy', 6, False),
        ('\\textbf{Energy Consumed (kWh)}', 'energy_consumed', 6, False),
        ('Water Consumed (L)', 'water_consumed', 6, False),
        ('Peak RAM Usage (GB)', 'ram_used_gb', 4, False)
    ]
    
    for label, key, decimals, is_gpu in metrics_config:
        values = [format_value(m[key], decimals, is_gpu) for m in all_metrics]
        values_str = " & ".join(values)
        
        # Apply bold if necessary
        if '\\textbf{' in label:
            values_str = " & ".join([f"\\textbf{{{v}}}" for v in values])
        
        print(f"    {label} & {values_str} \\\\")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    # Specify the CSV file paths here
    csv_files = [
        "warprec/experiments/Netflix100M-ItemKNN-Serial/codecarbon/emissions.csv",
        "warprec/experiments/Netflix100M-EASE-Serial/codecarbon/emissions.csv",
        "warprec/experiments/Netflix100M-NeuMF-Serial/codecarbon/emissions.csv",
        "warprec/experiments/Netflix100M-LightGCN-Serial/codecarbon/emissions.csv",
        "warprec/experiments/Netflix100M-SASRec-Serial/codecarbon/emissions.csv",
    ]
    
    generate_unified_latex_table(csv_files)
