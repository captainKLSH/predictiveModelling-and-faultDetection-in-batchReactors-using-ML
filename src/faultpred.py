import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# --- Configuration ---
# The paths to our original, trained tools
BASE_MODEL_PATH = 'reactor_fault_detector.h5'
PREPROCESSOR_PATH = 'preprocessor.pkl'

# Paths for our new process
NEW_DATA_DIR = 'esterification_data'
FINE_TUNED_MODEL_PATH = 'esterification_fault_detector.h5'


def generate_esterification_data():
    """
    Simulates and saves data for a new 'esterification' process.
    In a real project, you would replace this function with your actual data files.
    This function creates one normal batch and one faulty batch.
    """
    print("Simulating new data for the 'esterification' process...")
    if not os.path.exists(NEW_DATA_DIR):
        os.makedirs(NEW_DATA_DIR)

    # Create a base DataFrame (similar structure to our original data)
    time_steps = pd.to_timedelta(np.arange(0, 300 * 60, 30), unit='s')
    base_df = pd.DataFrame({'Time_min': time_steps.total_seconds() / 60})

    # --- Normal Batch Simulation ---
    normal_df = base_df.copy()
    normal_df['ReactorTemp'] = 350 + 15 * np.sin(np.pi * normal_df['Time_min'] / 300)
    normal_df['Pressure_bar'] = 2.5 - 0.5 * (normal_df['Time_min'] / 300)
    normal_df['Volume_m3'] = 10 + 10 * (normal_df['Time_min'] / 300)
    normal_df['Agitator_Speed_rpm'] = 600
    normal_df['Coolant_Flow_m3_s'] = 0.8
    normal_df['Feed_Flow_m3_s'] = 0.0015
    normal_df['Coolant_delatT_K'] = 8 + 2 * np.sin(np.pi * normal_df['Time_min'] / 300)
    normal_df['fault_type'] = 0  # Label as Normal
    normal_df.to_csv(os.path.join(NEW_DATA_DIR, 'ester_normal_batch_1.csv'), index=False)

    # --- Faulty Batch Simulation (e.g., Cooling Failure) ---
    fault_df = normal_df.copy()
    # Simulate a cooling failure after minute 150
    fault_mask = fault_df['Time_min'] > 150
    fault_df.loc[fault_mask, 'Coolant_Flow_m3_s'] = 0.05
    fault_df.loc[fault_mask, 'ReactorTemp'] += 20 * ((fault_df['Time_min'][fault_mask] - 150) / 150)
    fault_df.loc[fault_mask, 'fault_type'] = 1 # Label as Cooling Fault
    fault_df.to_csv(os.path.join(NEW_DATA_DIR, 'ester_cooling_fault_1.csv'), index=False)
    
    print(f"Generated new data in the '{NEW_DATA_DIR}' directory.")
    return [os.path.join(NEW_DATA_DIR, f) for f in os.listdir(NEW_DATA_DIR)]


def load_and_prep_new_data(preprocessor, file_paths):
    """
    Loads and preprocesses the new data using the OLD preprocessor.
    This ensures the data is scaled in the exact same way as the original data.
    """
    df_list = [pd.read_csv(f) for f in file_paths]
    master_df = pd.concat(df_list, ignore_index=True)

    # Use the preprocessor's scaler to transform the new data
    master_df[preprocessor.features] = preprocessor.scaler.transform(master_df[preprocessor.features])

    # Create sequences
    sequences, labels = [], []
    for i in range(len(master_df) - preprocessor.sequence_length):
        sequences.append(master_df[preprocessor.features].iloc[i:i + preprocessor.sequence_length].values)
        labels.append(master_df[preprocessor.label_column].iloc[i + preprocessor.sequence_length])
        
    X = np.array(sequences)
    # New process has 2 classes: Normal (0) and Cooling Fault (1)
    y = to_categorical(np.array(labels), num_classes=2)
    
    return X, y


if __name__ == '__main__':
    # Step 1: Generate simulated data for the new process
    new_data_files = generate_esterification_data()

    # Step 2: Load our original tools (the expert doctor and their equipment)
    print("\nLoading the base model and preprocessor...")
    base_model = tf.keras.models.load_model(BASE_MODEL_PATH)
    with open(PREPROCESSOR_PATH, 'rb') as f:
        preprocessor = pickle.load(f)

    # Step 3: Load and prepare the new data using the old preprocessor
    print("\nLoading and preparing new esterification data...")
    X_new, y_new = load_and_prep_new_data(preprocessor, new_data_files)
    X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.2, random_state=42)

    # Step 4: Perform the "fellowship" - modify the model for the new task
    print("\nModifying the model for transfer learning...")
    
    # Freeze the base model to preserve its core knowledge
    base_model.trainable = False
    
    # Create the new model architecture
    inputs = tf.keras.Input(shape=(preprocessor.sequence_length, len(preprocessor.features)))
    # We run the new data through the frozen base model
    x = base_model(inputs, training=False)
    # We add a new "decision-maker" layer for our new task
    # This new output layer has 2 neurons for "Normal" and "Cooling Fault"
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    
    transfer_model = tf.keras.Model(inputs, outputs)
    
    # Compile the new model. Use a low learning rate for fine-tuning.
    transfer_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
    
    print("New model architecture for fine-tuning:")
    transfer_model.summary()
    
    # Step 5: Fine-tune the model on the new data
    print("\nStarting fine-tuning (the 'fellowship' training)...")
    transfer_model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

    # Step 6: Evaluate the newly specialized model
    print("\nEvaluating the fine-tuned model...")
    loss, accuracy = transfer_model.evaluate(X_test, y_test)
    print(f"  -> Fine-tuned Model Accuracy: {accuracy * 100:.2f}%")

    # Step 7: Save the new expert model
    transfer_model.save(FINE_TUNED_MODEL_PATH)
    print(f"\nNew, specialized model saved to '{FINE_TUNED_MODEL_PATH}'")
