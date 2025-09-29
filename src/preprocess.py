import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from load_and_label import load_and_label_data
import pickle

# --- Configuration ---
SEQUENCE_LENGTH = 30
FEATURES = [
    'ReactorTemp', 'Pressure_bar', 'Volume_m3', 'Agitator_Speed_rpm',
    'Coolant_Flow_m3_s', 'Feed_Flow_m3_s', 'Coolant_delatT_K'
]
LABEL_COLUMN = 'fault_type'


class DataPreprocessor:
    """
    An object-oriented approach to preparing reactor data for an LSTM model.

    Think of this object as a specialized "prep station" in a kitchen.
    Once you build it, it remembers how you like your ingredients prepared
    (e.g., how you scaled the numbers). You can then reuse this same station
    to prepare new, incoming data in the exact same way, ensuring consistency.
    """
    def __init__(self, features, label_col, sequence_length):
        """
        Initializes the preprocessor with the specific settings for the data.
        """
        self.features = features
        self.label_col = label_col
        self.sequence_length = sequence_length
        # The scaler is created once and stored within the object.
        self.scaler = MinMaxScaler()

    def process(self, df):
        """
        Takes a raw DataFrame and performs all preprocessing steps.
        This method fits the scaler and transforms the data.
        """
        print(f"\nStarting preprocessing with sequence length = {self.sequence_length}...")
        
        # --- Step 1: Scale the Feature Data ---
        # The scaler learns the data's range and scales it.
        df[self.features] = self.scaler.fit_transform(df[self.features])
        print("  - Features scaled to a 0-1 range.")
        
        # --- Step 2: Create Sequences ---
        sequences = []
        labels = []
        
        # Group by source file to prevent creating sequences across different batches.
        for _, group in df.groupby('source_file'):
            feature_data = group[self.features].values
            label_data = group[self.label_col].values
            
            for i in range(len(group) - self.sequence_length):
                sequences.append(feature_data[i:i + self.sequence_length])
                labels.append(label_data[i + self.sequence_length])
                
        X = np.array(sequences)
        y = np.array(labels)
        print(f"  - Created {len(X)} sequences from the data.")

        # --- Step 3: One-Hot Encode Labels ---
        y = to_categorical(y, num_classes=3)
        print("  - Labels have been one-hot encoded.")
        print("\nPreprocessing complete!")
        
        return X, y

def save_preprocessor(preprocessor, path="pkl/preprocessor.pkl"):
    """
    Saves the preprocessor object (including its trained scaler) to a file.
    This is crucial for using the same scaling on new data later.
    """
    with open(path, 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"\nPreprocessor object saved to {path}")

if __name__ == '__main__':
    # Load the data using the script from our previous step.
    master_dataset = load_and_label_data()
    
    if master_dataset is not None:
        # --- Object 1: The Preprocessor ---
        # We create an instance of our preprocessor class. This is our first object.
        # It holds the configuration and the scaler.
        preprocessor_obj = DataPreprocessor(
            features=FEATURES,
            label_col=LABEL_COLUMN,
            sequence_length=SEQUENCE_LENGTH
        )
        
        # --- Object 2: The Processed Data ---
        # The .process() method returns a tuple containing our two main data objects:
        # the processed features (X) and processed labels (y).
        X_processed, y_processed = preprocessor_obj.process(master_dataset)
        
        # Save the preprocessor for later use (e.g., on live data)
        save_preprocessor(preprocessor_obj)
        
        # Verification of the processed data objects
        print("\n--- Verification ---")
        print(f"Shape of the processed features (X): {X_processed.shape}")
        print(f"Shape of the processed labels (y): {y_processed.shape}")
        
        print("\nExplanation of the shapes:")
        print(f" (Number of Samples, Sequence Length, Number of Features)")
        print(f" {X_processed.shape[0]} 'video clips' were created.")
        print(f" Each clip is {X_processed.shape[1]} time steps long.")
        print(f" Each time step has {X_processed.shape[2]} sensor readings.")
        
        print(f"\n (Number of Samples, Number of Classes)")
        print(f" We have {y_processed.shape[0]} labels, one for each clip.")
        print(f" Each label has {y_processed.shape[1]} possible categories (Normal, Agitator, Runaway).")

