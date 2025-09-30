import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

# --- Configuration ---
# The paths to the tools we saved in previous steps.
SAVED_MODEL_PATH = 'reactor_fault_detector.h5'
SAVED_PREPROCESSOR_PATH = 'preprocessor.pkl'
# A dictionary to translate the model's output into human-readable labels.
CLASS_NAMES = {0: 'Normal', 1: 'Agitator Fault', 2: 'Runaway Fault'}


def load_tools(model_path, preprocessor_path):
    """
    Loads the trained model and the data preprocessor from disk.
    
    From a non-technical view, this is like bringing our expert doctor (the model)
    and their specialized medical equipment (the preprocessor) into the clinic so
    they are ready to see a new patient.
    """
    print("Loading the trained AI model and preprocessor...")
    try:
        model = tf.keras.models.load_model(model_path)
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        print("Tools loaded successfully.")
        return model, preprocessor
    except FileNotFoundError:
        print(f"Error: Make sure '{model_path}' and '{preprocessor_path}' are in the correct directory.")
        return None, None


def prepare_new_data(preprocessor, df):
    """
    Prepares a new DataFrame of reactor data for prediction.
    This uses the *already fitted* scaler from our preprocessor.
    
    This is the 'triage' step. We take the new patient's raw data, use our
    pre-calibrated equipment to get the right measurements (scaling), and
    format it into a 'chart' (sequence) that our doctor can read.
    """
    # Scale the features using the saved scaler. DO NOT use .fit_transform here.
    df[preprocessor.features] = preprocessor.scaler.transform(df[preprocessor.features])
    
    # Create the sequence from the last `sequence_length` data points.
    # In a real-time system, you'd be feeding the latest data from the reactor.
    sequence = df[preprocessor.features].tail(preprocessor.sequence_length).values
    
    # The model expects a "batch" of data, so we add an extra dimension.
    return np.expand_dims(sequence, axis=0)
    

def diagnose_reactor(model, data_sequence, class_names):
    """
    Takes a prepared data sequence and returns the model's diagnosis.
    """
    # Get the raw predictions (probabilities for each class).
    predictions = model.predict(data_sequence)
    
    # Find the class with the highest probability.
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class_index] * 100
    
    # Get the human-readable name for that class.
    diagnosis = class_names[predicted_class_index]
    
    return diagnosis, confidence


if __name__ == '__main__':
    # Step 1: Load our saved model and preprocessor
    model, preprocessor = load_tools(SAVED_MODEL_PATH, SAVED_PREPROCESSOR_PATH)
    
    if model and preprocessor:
        # Step 2: Load a sample of new data to test the system.
        # We'll use one of the original fault files as a "new" patient.
        # Let's see if it can correctly diagnose an agitator fault in progress.
        try:
            # We load the full file, but the function will only use the most recent
            # data points to simulate a real-time prediction.
            new_patient_data = pd.read_csv('fault_data/agitator_fault.csv')
            print("\nSuccessfully loaded new data sample for diagnosis.")
            
            # Step 3: Prepare the new data for the model.
            processed_sequence = prepare_new_data(preprocessor, new_patient_data)
            
            # Step 4: Get the diagnosis from the AI.
            diagnosis, confidence = diagnose_reactor(model, processed_sequence, CLASS_NAMES)
            
            # --- Display the final result ---
            print("\n--- REACTOR DIAGNOSIS ---")
            print(f"      Status: {diagnosis}")
            print(f"  Confidence: {confidence:.2f}%")
            print("-------------------------")

        except FileNotFoundError:
            print("\nError: Could not find 'fault_data/agitator_fault.csv'.")
            print("Please run 'generate_fault_data.py' to create the sample data.")
