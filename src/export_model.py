"""
Model Export Module
-------------------

This script trains the neural network on the full Heart Disease dataset
and exports the learned parameters (weights and biases) to a JSON file.

The goal is to:
- Train a stable model once.
- Serialize its parameters in a portable and human-readable format.
- Allow other scripts (e.g. import_model.py or a Streamlit app) to
  load the trained parameters and perform inference without retraining.

This is similar in spirit to how frameworks like PyTorch or TensorFlow
save and load model checkpoints, but implemented manually for an
educational, from-scratch neural network.
"""

import json
from pathlib import Path

import numpy as np

from data_cleaning import X, y
from model_components import (
    DenseLayer,
    ActivationReLU,
    Dropout,
    ActivationSoftmaxLossCategoricalCrossentropy,
    OptimizerAdam,
)


def train_model_full_batch(
    X_data: np.ndarray,
    y_data: np.ndarray,
    hidden_units: int = 16,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
    decay: float = 1e-3,
    epochs: int = 500,
    random_seed: int = 42,
):
    """
    Train the neural network on the full dataset using full-batch gradient descent.

    Parameters
    ----------
    X_data : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y_data : np.ndarray
        Target vector of shape (n_samples,), containing class indices (0 or 1).
    hidden_units : int
        Number of neurons in the hidden dense layer.
    dropout_rate : float
        Dropout rate applied after the hidden layer (fraction of dropped neurons).
    learning_rate : float
        Initial learning rate for the Adam optimizer.
    decay : float
        Learning rate decay factor for Adam.
    epochs : int
        Number of training epochs.
    random_seed : int
        Seed for NumPy's RNG to ensure reproducibility.

    Returns
    -------
    dict
        A dictionary containing:
        - "layer1": trained DenseLayer instance for the hidden layer
        - "layer2": trained DenseLayer instance for the output layer
        - "history": training history with loss and accuracy per epoch
    """
    np.random.seed(random_seed)

    n_features = X_data.shape[1]
    n_classes = 2

    # Instantiate model components
    layer1 = DenseLayer(n_inputs=n_features, n_neurons=hidden_units)
    activation1 = ActivationReLU()
    dropout1 = Dropout(rate=dropout_rate)
    layer2 = DenseLayer(n_inputs=hidden_units, n_neurons=n_classes)
    loss_activation = ActivationSoftmaxLossCategoricalCrossentropy()
    optimizer = OptimizerAdam(learning_rate=learning_rate, decay=decay)

    # Containers for training history (optional, but nice to export)
    losses = []
    accuracies = []

    print(
        f"Starting full-batch training on {X_data.shape[0]} samples "
        f"with {n_features} features, {hidden_units} hidden units."
    )

    for epoch in range(epochs):
        # Forward pass
        layer1.forward(X_data)
        activation1.forward(layer1.output)
        dropout1.forward(activation1.output, training=True)
        layer2.forward(dropout1.output)
        loss = loss_activation.forward(layer2.output, y_data)

        # Predictions & accuracy
        predictions = np.argmax(loss_activation.output, axis=1)
        accuracy = np.mean(predictions == y_data)

        losses.append(float(loss))
        accuracies.append(float(accuracy))

        # Backward pass
        loss_activation.backward(loss_activation.output, y_data)
        layer2.backward(loss_activation.dinputs)
        dropout1.backward(layer2.dinputs)
        activation1.backward(dropout1.dinputs)
        layer1.backward(activation1.dinputs)

        # Parameter update
        optimizer.pre_update_params()
        optimizer.update_params(layer1)
        optimizer.update_params(layer2)
        optimizer.post_update_params()

        # Logging every 50 epochs
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"Loss: {loss:.4f} | Accuracy: {accuracy:.4f} | "
                f"LR: {optimizer.current_learning_rate:.6f}"
            )

    print("Training completed.")

    return {
        "layer1": layer1,
        "layer2": layer2,
        "history": {"loss": losses, "accuracy": accuracies},
        "hyperparams": {
            "hidden_units": hidden_units,
            "dropout_rate": dropout_rate,
            "learning_rate": learning_rate,
            "decay": decay,
            "epochs": epochs,
            "random_seed": random_seed,
        },
    }


def export_model_to_json(
    layer1: DenseLayer,
    layer2: DenseLayer,
    history: dict,
    hyperparams: dict,
    output_path: Path,
):
    """
    Export trained model parameters and metadata to a JSON file.

    Parameters
    ----------
    layer1 : DenseLayer
        Trained hidden dense layer.
    layer2 : DenseLayer
        Trained output dense layer.
    history : dict
        Training history (loss and accuracy per epoch).
    hyperparams : dict
        Hyperparameters used during training.
    output_path : Path
        Path where the JSON file will be written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build a portable representation of the model
    model_export = {
        "model_type": "neural_network",
        "problem_type": "binary_classification",
        "architecture": {
            "input_dim": int(layer1.weights.shape[0]),
            "hidden_units": int(layer1.weights.shape[1]),
            "output_dim": int(layer2.weights.shape[1]),
            "activation_hidden": "ReLU",
            "activation_output": "Softmax",
        },
        "training": {
            "loss_history": history["loss"],
            "accuracy_history": history["accuracy"],
            **hyperparams,
        },
        "parameters": {
            "layer1": {
                "weights": layer1.weights.tolist(),
                "biases": layer1.biases.tolist(),
            },
            "layer2": {
                "weights": layer2.weights.tolist(),
                "biases": layer2.biases.tolist(),
            },
        },
    }

    # Write a pretty (human-readable) JSON
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(model_export, f, indent=4)

    print(f"\nModel parameters exported successfully to: {output_path}")


def main():
    """
    Entry point for training and exporting the model.

    This function:
    - Loads the cleaned dataset (X, y)
    - Trains the neural network on the full dataset
    - Exports the trained parameters and training metadata to JSON
    """
    # Convert pandas objects to NumPy arrays
    X_data = X.values
    y_data = y.values

    # Train model
    result = train_model_full_batch(
        X_data=X_data,
        y_data=y_data,
        hidden_units=16,
        dropout_rate=0.2,
        learning_rate=0.001,
        decay=1e-3,
        epochs=500,
        random_seed=42,
    )

    layer1 = result["layer1"]
    layer2 = result["layer2"]
    history = result["history"]
    hyperparams = result["hyperparams"]

    # Define output path (e.g. project_root/models/heart_disease_model.json)
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    output_path = models_dir / "heart_disease_nn_model.json"

    # Export to JSON
    export_model_to_json(
        layer1=layer1,
        layer2=layer2,
        history=history,
        hyperparams=hyperparams,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()