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


def load_and_prep_new_data(preprocessor,fp):
    """
    Loads and preprocesses the new data using the OLD preprocessor.
    This ensures the data is scaled in the exact same way as the original data.
    """
    #df_list = [pd.read_csv(f) for f in file_paths]
    master_df = pd.read_csv(fp,header=0)

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
