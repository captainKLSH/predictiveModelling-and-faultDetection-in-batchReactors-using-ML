import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
# --- Configuration ---
# The file paths for the data we created in the previous step.
PROCESSED_DATA_X_PATH = 'X_processed.npy'
PROCESSED_DATA_Y_PATH = 'y_processed.npy'
# The filename for our final, trained AI model.
SAVED_MODEL_PATH = 'reactor_fault_detector.h5'


def build_model(input_shape):
    """
    Builds the LSTM neural network architecture.

    From a non-technical view, this is the blueprint for our AI's brain.
    - The LSTM layers are like the 'memory centers', specialized in understanding
      the story or sequence of our reactor data.
    - The Dropout layer is a clever trick to prevent the AI from 'memorizing'
      the training data too perfectly. It's like a teacher telling a student
      not to rely on just one way to solve a problem.
    - The final Dense layer is the 'decision-maker'. It looks at the analysis
      from the memory layers and makes the final call: Normal, Agitator Fault,
      or Runaway Fault.
    """
    model = Sequential([
        # Input layer specifies the shape of our 'video clips'
        # (20 time steps, 7 features each).
        tf.keras.Input(shape=input_shape),

        # First LSTM layer with 64 memory units. 'return_sequences=True' tells it
        # to pass its full analysis to the next layer, not just its final thought.
        LSTM(64, return_sequences=True),
        Dropout(0.2),  # Dropout for regularization

        # Second LSTM layer. This one only passes its final summary.
        LSTM(32),
        Dropout(0.2),

        # The final decision-making layer. It has 3 neurons, one for each
        # of our classes. The 'softmax' function ensures the outputs are
        # probabilities that sum to 1.
        Dense(3, activation='softmax')
    ])

    # The 'compile' step assembles the model with its learning tools.
    # - 'optimizer='adam'' is an efficient algorithm for learning.
    # - 'loss='categorical_crossentropy'' is the math formula used to
    #   calculate how wrong the model's predictions are.
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print("Model architecture has been built successfully.")
    model.summary()
    return model

def plot_training_history(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()
def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Visualizes the model's performance with a confusion matrix.
    This chart shows us exactly where the model is getting things right
    and where it's getting confused.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Fault Type')
    plt.xlabel('Predicted Fault Type')
    plt.show()

def evaluate_model(model, X_test, y_test):
    # Predict classes for the test set
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Calculate precision, recall, f1-score
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
if __name__ == '__main__':
    # --- Step 1: Load the Preprocessed Data ---
    print("Loading preprocessed data...")
    X = np.load(PROCESSED_DATA_X_PATH)
    y = np.load(PROCESSED_DATA_Y_PATH)

    # --- Step 2: Split Data into Training and Testing Sets ---
    # We set aside 20% of the data for the 'final exam' (testing).
    # 'stratify=y' ensures that both the training and testing sets get a
    # proportional mix of normal, agitator, and runaway examples.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Data split complete: {len(X_train)} for training, {len(X_test)} for testing.")

    # --- Step 3: Build the LSTM Model ---
    # The input shape is (sequence_length, num_features)
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape)

    # --- Step 4: Train the Model ---
    print("\nStarting model training...")
    # 'epochs=10' means the model will go through the entire training dataset 10 times.
    # 'batch_size=32' means it looks at 32 samples at a time.
    # 'validation_split=0.1' uses 10% of the training data to check progress after each epoch.
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    print("Model training complete.")

    # --- Step 5: Evaluate the Model ---
    print("\nEvaluating model on the unseen test data...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"  -> Test Accuracy: {accuracy * 100:.2f}%")
    print(f"  -> Test Loss: {loss:.4f}")

    # Plot training and validation performance
    plot_training_history(history)

    # Evaluate precision, recall, F1 score on test data
    evaluate_model(model, X_test, y_test)

    # --- Step 6: Analyze with a Confusion Matrix ---
    print("\nGenerating confusion matrix...")
    # Get the model's predictions on the test set.
    y_pred_probs = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    class_names = ['Normal', 'Agitator Fault', 'Runaway Fault']
    plot_confusion_matrix(y_true_classes, y_pred_classes, class_names)

    # --- Step 7: Save the Trained Model ---
    model.save(SAVED_MODEL_PATH)
    print(f"\nTrained model saved successfully to '{SAVED_MODEL_PATH}'")
