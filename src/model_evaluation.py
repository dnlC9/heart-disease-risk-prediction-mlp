import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# Importamos los datos ya limpios (X, y) desde el módulo de cleaning
from data_cleaning import X, y

# Importamos los componentes del modelo ya entrenable
from model_components import (
    DenseLayer,
    ActivationReLU,
    ActivationSoftmaxLossCategoricalCrossentropy,
    OptimizerAdam,
)

print(f"Dataset shape -> X: {X.shape}, y: {y.shape}")

# 1) Definimos el número de divisiones (folds)
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# 2) Recorremos los folds y mostramos las dimensiones de train/test
fold_id = 0
all_test_indices = []

for train_index, test_index in kf.split(X):
    fold_id += 1

    # Si X e y son DataFrame/Series de pandas, usamos .iloc
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    print(f"\nFold {fold_id}/{n_splits}")
    print(f"  X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"  X_test  shape: {X_test.shape}, y_test  shape: {y_test.shape}")

    # Guardamos los índices de test para comprobar la cobertura total al final
    all_test_indices.extend(test_index)

# 3) Comprobación de que el dataset se ha dividido en 5 partes sin solaparse
all_test_indices = np.array(all_test_indices)

print("\n--- K-Fold split check ---")
print(f"Total samples in dataset:      {X.shape[0]}")
print(f"Total test indices seen:       {len(all_test_indices)}")
print(f"Unique test indices:           {len(np.unique(all_test_indices))}")

# Checks de integridad (profesional)
assert len(all_test_indices) == X.shape[0], \
    "Some samples were not included in any test fold."

assert len(np.unique(all_test_indices)) == X.shape[0], \
    "Some samples appear multiple times in test folds."

print("K-Fold integrity check passed ✅")


# ----------------------------------------------------------
#         TRAIN & EVALUATE MODEL IN EACH K-FOLD
# ----------------------------------------------------------

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Lists to store metrics per fold
fold_accuracies = []
fold_sensitivities = []   # recall for class "1" (disease)
fold_specificities = []   # recall for class "0" (no disease)
fold_precisions = []
fold_f1s = []
fold_aucs = []

epochs = 300

print("\nStarting K-Fold training and evaluation...\n")

fold_id = 0

for train_index, test_index in kf.split(X):
    fold_id += 1
    print(f"\n========== Fold {fold_id}/{n_splits} ==========")

    # Split data
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Convert to NumPy arrays (our NN works with NumPy)
    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values

    n_features = X_train.shape[1]
    n_classes = 2

    # Re-instantiate model for each fold (very important)
    layer1 = DenseLayer(n_inputs=n_features, n_neurons=16)
    activation1 = ActivationReLU()
    layer2 = DenseLayer(n_inputs=16, n_neurons=n_classes)
    loss_activation = ActivationSoftmaxLossCategoricalCrossentropy()
    optimizer = OptimizerAdam(learning_rate=0.001, decay=1e-3)

    # -------- TRAINING LOOP FOR THIS FOLD --------
    for epoch in range(epochs):
        # Forward pass
        layer1.forward(X_train)
        activation1.forward(layer1.output)
        layer2.forward(activation1.output)
        loss = loss_activation.forward(layer2.output, y_train)

        # Predictions & accuracy (train)
        train_predictions = np.argmax(loss_activation.output, axis=1)
        train_accuracy = np.mean(train_predictions == y_train)

        # Backward pass
        loss_activation.backward(loss_activation.output, y_train)
        layer2.backward(loss_activation.dinputs)
        activation1.backward(layer2.dinputs)
        layer1.backward(activation1.dinputs)

        # Update weights
        optimizer.pre_update_params()
        optimizer.update_params(layer1)
        optimizer.update_params(layer2)
        optimizer.post_update_params()

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(
                f"  Epoch {epoch}/{epochs} | "
                f"Train loss: {loss:.4f} | Train acc: {train_accuracy:.4f}"
            )

    # -------- EVALUATION ON TEST SPLIT --------
    # Forward pass on test data
    layer1.forward(X_test)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    test_loss = loss_activation.forward(layer2.output, y_test)

    # Predicted class labels
    test_predictions = np.argmax(loss_activation.output, axis=1)
    test_accuracy = np.mean(test_predictions == y_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, test_predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Sensitivity (recall for positive class = 1)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Specificity (recall for negative class = 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Precision, recall, F1 (for class 1)
    precision = precision_score(y_test, test_predictions, zero_division=0)
    recall = recall_score(y_test, test_predictions, zero_division=0)
    f1 = f1_score(y_test, test_predictions, zero_division=0)

    # ROC-AUC (use probability of class 1 from Softmax)
    y_proba_class1 = loss_activation.output[:, 1]
    auc = roc_auc_score(y_test, y_proba_class1)

    print(f"  Test loss: {test_loss:.4f} | Test acc: {test_accuracy:.4f}")
    print(f"  Confusion matrix:\n{cm}")
    print(f"  Sensitivity (recall class 1): {sensitivity:.4f}")
    print(f"  Specificity (class 0):        {specificity:.4f}")
    print(f"  Precision:                    {precision:.4f}")
    print(f"  F1-score:                     {f1:.4f}")
    print(f"  ROC-AUC:                      {auc:.4f}")

    # Store metrics for this fold
    fold_accuracies.append(test_accuracy)
    fold_sensitivities.append(sensitivity)
    fold_specificities.append(specificity)
    fold_precisions.append(precision)
    fold_f1s.append(f1)
    fold_aucs.append(auc)

# -------- SUMMARY ACROSS FOLDS --------
def summarize_metric(name, values):
    mean = np.mean(values)
    std = np.std(values)
    print(f"{name}: {mean:.4f} ± {std:.4f}")

print("\n========== Cross-Validation Summary ==========")
summarize_metric("Accuracy", fold_accuracies)
summarize_metric("Sensitivity (recall class 1)", fold_sensitivities)
summarize_metric("Specificity (class 0)", fold_specificities)
summarize_metric("Precision", fold_precisions)
summarize_metric("F1-score", fold_f1s)
summarize_metric("ROC-AUC", fold_aucs)