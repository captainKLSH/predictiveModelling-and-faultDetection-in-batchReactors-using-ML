import pandas as pd
import os

DATA_DIR = 'raw'
SUB_DIR1='normal'
SUB_DIR2='faults'
NORMAL_FILES = [f'normalbatch{i}.csv' for i in range(1, 8)]
# These files MUST have a 'fault_type' column added manually.
FAULT_FILES = ['agitatorfault_L.csv', 'runawayfault_L.csv']

def load_and_label_data():
    """
    Loads all reactor batch data, using pre-existing labels for fault files,
    and combines it into a single master DataFrame.

    This script now acts as an "assembler." For the normal files, it will
    automatically label them. For the fault files, it trusts that you have
    already added a 'fault_type' column and will use those labels directly.
    """
    all_dataframes = []
    
    # --- Step 1: Load and Auto-Label Normal Batches ---
    # We'll start by reading all the "normal" files. We'll add the
    # 'fault_type' column and label every row as '0' (Normal).
    print("Loading and auto-labeling normal batch files...")
    for filename in NORMAL_FILES:
        filepath = os.path.join(DATA_DIR,SUB_DIR1, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['fault_type'] = 0  # Label '0' for Normal
            df['source_file'] = filename # Keep track of where the data came from
            all_dataframes.append(df)
            print(f"  - Loaded and labeled {filename} as Normal.")
        else:
            print(f"  - WARNING: {filename} not found.")

    # --- Step 2: Load Manually Labeled Fault Batches ---
    # For the fault files, the script will check for a 'fault_type' column.
    # It's your job to create this column in your CSVs and label the rows
    # with 0 (Normal), 1 (Agitator Fault), or 2 (Runaway Fault).
    print("\nLoading manually labeled fault batch files...")
    for filename in FAULT_FILES:
        filepath = os.path.join(DATA_DIR,SUB_DIR2, filename)
        
        if not os.path.exists(filepath):
            print(f"  - WARNING: {filename} not found.")
            continue
            
        df = pd.read_csv(filepath)
        
        # IMPORTANT: Check if the user has added the 'fault_type' column.
        if 'fault_type' not in df.columns:
            print(f"  - ERROR: The file '{filename}' is missing the required 'fault_type' column.")
            print("    Please open this CSV and add a 'fault_type' column with your manual labels (0, 1, or 2).")
            print("    Skipping this file for now.")
            continue

        print(f"  - Loaded {filename} using your manual labels.")
        df['source_file'] = filename
        all_dataframes.append(df)
        
    # --- Step 3: Combine into a Master Dataset ---
    # Finally, we take all our individually loaded tables and stack them
    # into one giant master table.
    if not all_dataframes:
        print("\nNo data was loaded. Please check file paths and ensure fault files are labeled.")
        return None
        
    master_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"\nSuccessfully combined all data into a master table with {len(master_df)} rows.")
    
    return master_df