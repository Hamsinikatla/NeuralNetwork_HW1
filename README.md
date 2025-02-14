# NeuralNetwork_HW1

700772580
Hamsini katla 

# 1. Tensor Manipulations & Reshaping

## Overview
This walkthrough demonstrates essential tensor manipulations using TensorFlow. It covers:
- Creating a random tensor.
- Finding its rank and shape.
- Reshaping and transposing the tensor.
- Broadcasting and performing element-wise addition with a smaller tensor.

## Installation Instructions
To run this script, ensure you have TensorFlow installed. You can install it using:
```bash
pip install tensorflow
```

## Usage Guide
1. Save the script as `tensor_operations.py`.
2. Run the script using:
```bash
python tensor_operations.py
```

## Code Explanation
### Step 1: Create a Random Tensor
```python
import tensorflow as tf
tensor = tf.random.uniform(shape=(4, 6))
print("Original Tensor:\n", tensor.numpy())
```
- Generates a random tensor of shape (4, 6) with values between 0 and 1.
- `tf.random.uniform()` creates tensors with random values.

### Step 2: Find Rank and Shape
```python
rank = tf.rank(tensor)
shape = tf.shape(tensor)
print(f"\nOriginal Rank: {rank.numpy()}, Original Shape: {shape.numpy()}")
```
- `tf.rank(tensor)`: Returns the rank (number of dimensions) of the tensor.
- `tf.shape(tensor)`: Returns the shape of the tensor.

### Step 3: Reshape and Transpose
```python
reshaped_tensor = tf.reshape(tensor, (2, 3, 4))
print(f"\nReshaped Shape: {reshaped_tensor.shape}")
```
- Reshapes the tensor from (4, 6) to (2, 3, 4).
- The total number of elements remains the same.

```python
transposed_tensor = tf.transpose(reshaped_tensor, perm=[1, 0, 2])
print(f"Transposed Shape: {transposed_tensor.shape}")
```
- Transposes the tensor, changing its shape from (2, 3, 4) to (3, 2, 4).
- `perm=[1, 0, 2]` swaps the first and second dimensions.

### Step 4: Broadcasting and Addition
```python
small_tensor = tf.random.uniform(shape=(1, 4))
broadcasted_result = transposed_tensor + small_tensor
print("\nResult after Broadcasting and Addition:\n", broadcasted_result.numpy())
```
- Creates a smaller tensor of shape (1, 4).
- **Broadcasting**: TensorFlow automatically expands `small_tensor` to match the shape of `transposed_tensor` for element-wise addition.

## Output
The script prints:
- The original tensor
- Its rank and shape
- The reshaped and transposed tensor shapes
- The result after broadcasting and addition

### Summary of Outputs
1. **Rank & Shape before Reshaping**: Rank: 2, Shape: (4,6)
2. **After Reshaping**: Shape: (2,3,4)
3. **After Transposing**: Shape: (3,2,4)
4. **Broadcasting Happens**: (1,4) → (3,2,4), allowing element-wise addition.

## Key Learnings and Features
- **Reshaping** allows us to change tensor structures while preserving elements.
- **Transposing** swaps dimensions to rearrange data.
- **Broadcasting** automatically expands smaller tensors to match larger tensors for operations.










# 2.	Loss Functions & Hyperparameter Tuning

# Loss Functions: Mean Squared Error (MSE) & Categorical Cross-Entropy (CCE)

## Overview
This program demonstrates, implements, calculates, and compares two common loss functions:
- **Mean Squared Error (MSE)** – Used for regression tasks.
- **Categorical Cross-Entropy (CCE)** – Used for classification tasks.

The program modifies predictions slightly and plots a bar chart to compare how the loss values change. It also visualizes how these loss values change when model predictions are modified. The goal is to compute these loss values for different predictions and analyze how slight modifications in predictions affect the loss values. Additionally, the results are visualized using a bar chart.

## Installation Instructions
To run this script, ensure you have the necessary dependencies installed. You can install them using:
```bash
pip install tensorflow numpy matplotlib
```

## Usage Guide
1. Save the script as `loss_comparison.py`.
2. Run the script using:
```bash
python loss_comparison.py
```

## Code Explanation
### Step 1: Define True Values and Model Predictions
```python
y_true = tf.constant([[0, 0, 1], [1, 0, 0]], dtype=tf.float32)  # One-hot encoded labels
y_pred_1 = tf.constant([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]], dtype=tf.float32)  # Prediction 1
y_pred_2 = tf.constant([[0.2, 0.3, 0.5], [0.7, 0.2, 0.1]], dtype=tf.float32)  # Prediction 2
```
- `y_true` represents the true class labels in a one-hot encoded format.
- `y_pred_1` and `y_pred_2` are two different sets of predictions.

### Step 2: Compute Mean Squared Error (MSE) Loss
```python
mse_loss_1 = tf.keras.losses.MeanSquaredError()(y_true, y_pred_1).numpy()
mse_loss_2 = tf.keras.losses.MeanSquaredError()(y_true, y_pred_2).numpy()
```
- MSE is calculated between `y_true` and both sets of predictions.

### Step 3: Compute Categorical Cross-Entropy (CCE) Loss
```python
cce_loss_1 = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred_1).numpy()
cce_loss_2 = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred_2).numpy()
```
- CCE is computed for both predictions to measure how well they align with the true labels.

### Step 4: Print and Visualize Loss Values
```python
labels = ["MSE (Pred 1)", "MSE (Pred 2)", "CCE (Pred 1)", "CCE (Pred 2)"]
loss_values = [mse_loss_1, mse_loss_2, cce_loss_1, cce_loss_2]

plt.figure(figsize=(8,5))
plt.bar(labels, loss_values, color=['blue', 'lightblue', 'red', 'pink'])
plt.xlabel("Loss Type")
plt.ylabel("Loss Value")
plt.title("Comparison of MSE and CCE Losses")
plt.show()
```
- A bar chart is created to visually compare the loss values for MSE and CCE.

## Output
- The script prints the loss values for both MSE and CCE for each set of predictions.
- A bar chart visualizing the differences in loss values is displayed.

### Key Learnings and Features
- **Loss Functions**: Comparison of MSE and CCE in different prediction scenarios.
- **TensorFlow Implementation**: How to use TensorFlow to compute loss values.
- **Visualization**: Matplotlib is used to compare loss values effectively.

#### **Bar Chart Representation**
- **Blue shades** → MSE losses
- **Red shades** → CCE losses
- Shows how loss values change with different predictions.

## How to Run the Code
1. Save the script as `loss_comparison.py`.
2. Run the script using:
```bash
python loss_comparison.py
```
3. The loss values will be printed, and a bar chart will be displayed.

## Conclusion
This project illustrates the importance of choosing the right loss function for a given machine learning problem. MSE works better for regression, while CCE is crucial for classification tasks.










# 3.Train a Model with Different Optimizers

# Comparison of Adam vs. SGD Optimizers on Digits Dataset

## Overview
This project implements and compares two optimization techniques, **Adam** and **SGD**, for training an MLP classifier on the MNIST-like Digits dataset. The goal is to evaluate the effectiveness of these optimizers by analyzing training and validation accuracy trends.

## Installation Instructions
To run this script, ensure you have the necessary dependencies installed. You can install them using:
```bash
pip install numpy matplotlib scikit-learn
```

## Usage Guide
1. Save the script as `optimizer_comparison.py`.
2. Run the script using:
```bash
python optimizer_comparison.py
```
3. The script will train the models, compute accuracy, and display a bar chart comparing the performance of Adam and SGD.

## Code Explanation
### Step 1: Load and Preprocess the Data
```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

digits = load_digits()
X, y = digits.data / 16.0, digits.target  # Normalize data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
- Loads the MNIST-like Digits dataset.
- Normalizes and standardizes the data.
- Splits the dataset into training and testing sets.

### Step 2: Train MLP Models with Adam and SGD Optimizers
```python
from sklearn.neural_network import MLPClassifier

adam_model = MLPClassifier(hidden_layer_sizes=(128,), activation='relu', solver='adam', max_iter=20, random_state=42)
sgd_model = MLPClassifier(hidden_layer_sizes=(128,), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=20, random_state=42)

adam_model.fit(X_train, y_train)
sgd_model.fit(X_train, y_train)
```
- Creates two MLP classifiers using **Adam** and **SGD** optimizers.
- Trains both models on the dataset.

### Step 3: Compute Training and Validation Accuracy
```python
from sklearn.metrics import accuracy_score

adam_train_acc = accuracy_score(y_train, adam_model.predict(X_train))
adam_val_acc = accuracy_score(y_test, adam_model.predict(X_test))
sgd_train_acc = accuracy_score(y_train, sgd_model.predict(X_train))
sgd_val_acc = accuracy_score(y_test, sgd_model.predict(X_test))
```
- Evaluates training and validation accuracy for both models.

### Step 4: Compare Training and Validation Accuracy Trends
```python
import matplotlib.pyplot as plt

labels = ['Adam Train', 'Adam Val', 'SGD Train', 'SGD Val']
values = [adam_train_acc, adam_val_acc, sgd_train_acc, sgd_val_acc]

plt.figure(figsize=(8, 5))
plt.bar(labels, values, color=['blue', 'orange', 'blue', 'orange'])
plt.ylabel('Accuracy')
plt.title('Comparison of Adam vs. SGD on Digits Dataset')
plt.show()
```
- Creates a bar chart to visualize the accuracy comparison between Adam and SGD optimizers.

## Output
- The script prints training and validation accuracy for both optimizers.
- A bar chart is displayed, comparing Adam and SGD performance.

## Summary of Outputs
1. **Adam Optimizer:**
   - Higher validation accuracy, indicating better generalization.
   - Fast convergence.
2. **SGD Optimizer:**
   - May overfit or underperform depending on learning rate.
   - Slower convergence compared to Adam.

## Key Learnings and Features
- **Adam generally outperforms SGD** in terms of validation accuracy.
- **SGD may require careful tuning** of the learning rate for optimal performance.
- **Data Preprocessing Matters:** Standardization improves the training process.
- **Bar Chart Visualization** helps compare optimizer performance effectively.

## Conclusion
This project demonstrates how different optimization techniques affect model performance. Adam provides better results with minimal tuning, whereas SGD may require hyperparameter adjustments for competitive performance. The visualization aids in understanding the effectiveness of each optimizer.







# 4. Train a Neural Network and Log to TensorBoard








