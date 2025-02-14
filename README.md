# NeuralNetwork_HW1

700772580
Hamsini katla 


1.	Tensor Manipulations & Reshaping.

In this walkthrough, I'll explain the TensorFlow code step by step. 
Overview 
This demonstrates essential tensor manipulations using TensorFlow. It covers:
•	Creating a random tensor.
•	Finding its rank and shape.
•	Reshaping and transposing the tensor.
•	Broadcasting and performing element-wise addition with a smaller tensor.
Installation Instructions
To run this script, ensure you have TensorFlow installed. You can install it using:
Pip install tensorflow
Usage Guide
1.	Save the script as tensor_operations.py.
2.	Run the script using:
python tensor_operations.py

Code Explanation
Step 1: Create a Random Tensor
import tensorflow as tf
tensor = tf.random.uniform(shape=(4, 6))
print("Original Tensor:\n", tensor.numpy())
•	Generates a random tensor of shape (4, 6) with values between 0 and 1.
•	tf.random.uniform() creates tensors with random values.
Step 2: Find Rank and Shape
rank = tf.rank(tensor)
shape = tf.shape(tensor)
print(f"\nOriginal Rank: {rank.numpy()}, Original Shape: {shape.numpy()}")
•	tf.rank(tensor): Returns the rank (number of dimensions) of the tensor.
•	tf.shape(tensor): Returns the shape of the tensor.
Step 3: Reshape and Transpose
reshaped_tensor = tf.reshape(tensor, (2, 3, 4))
print(f"\nReshaped Shape: {reshaped_tensor.shape}")
•	Reshapes the tensor from (4, 6) to (2, 3, 4).
•	The total number of elements remains the same.
transposed_tensor = tf.transpose(reshaped_tensor, perm=[1, 0, 2])
print(f"Transposed Shape: {transposed_tensor.shape}")
•	Transposes the tensor, changing its shape from (2, 3, 4) to (3, 2, 4).
•	perm=[1, 0, 2] swaps the first and second dimensions.
Step 4: Broadcasting and Addition
small_tensor = tf.random.uniform(shape=(1, 4))
broadcasted_result = transposed_tensor + small_tensor
print("\nResult after Broadcasting and Addition:\n", broadcasted_result.numpy())
•	Creates a smaller tensor of shape (1, 4).
•	Broadcasting: TensorFlow automatically expands small_tensor to match the shape of transposed_tensor for element-wise addition.
 Output
•	The script prints:
o	The original tensor
o	Its rank and shape
o	The reshaped and transposed tensor shapes
o	The result after broadcasting and addition

Summary of Outputs

1.	Rank & Shape before Reshaping: (Rank: 2, Shape: (4,6))
2.	After Reshaping: (Shape: (2,3,4))
3.	After Transposing: (Shape: (3,2,4))
4.	Broadcasting Happens: (1,4) → (3,2,4), allowing element-wise addition.

Key learnings and features
•	Reshaping allows us to change tensor structures while preserving elements.
•	Transposing swaps dimensions to rearrange data.
•	Broadcasting automatically expands smaller tensors to match larger tensors for operations.



2.	Loss Functions & Hyperparameter Tuning
         Overview

This program demonstrates, implements, calculates and compares two common loss functions: Mean Squared Error (MSE) and Categorical Cross-Entropy (CCE) in TensorFlow. 
Mean Squared Error (MSE) – Used for regression tasks.
Categorical Cross-Entropy (CCE) – Used for classification tasks.
The program also modifies predictions slightly and plots a bar chart to compare how the loss values change. It also visualizes how these loss values change when model predictions are modified. The goal is to compute these loss values for different predictions and analyze how slight modifications in predictions affect the loss values. Additionally, the results are visualized using a bar chart.
Installation Instructions
To run this script, ensure you have the necessary dependencies installed. You can install them using:
pip install tensorflow numpy matplotlib
Usage Guide
1.	Save the script as loss_functions.py.
2.	Run the script using:
python loss_functions.py
Code Explanation
Step 1: Define True Values and Model Predictions
y_true = tf.constant([[0, 0, 1], [1, 0, 0]], dtype=tf.float32)  # One-hot encoded labels
y_pred_1 = tf.constant([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]], dtype=tf.float32)  # Prediction 1
y_pred_2 = tf.constant([[0.2, 0.3, 0.5], [0.7, 0.2, 0.1]], dtype=tf.float32)  # Prediction 2
•	y_true represents the true class labels in a one-hot encoded format.
•	y_pred_1 and y_pred_2 are two different sets of predictions.
Step 2: Compute Mean Squared Error (MSE) Loss
mse_loss_1 = tf.keras.losses.MeanSquaredError()(y_true, y_pred_1).numpy()
mse_loss_2 = tf.keras.losses.MeanSquaredError()(y_true, y_pred_2).numpy()
•	MSE is calculated between y_true and both sets of predictions.
Step 3: Compute Categorical Cross-Entropy (CCE) Loss
cce_loss_1 = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred_1).numpy()
cce_loss_2 = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred_2).numpy()
•	CCE is computed for both predictions to measure how well they align with the true labels.
Step 4: Print and Visualize Loss Values
labels = ["MSE (Pred 1)", "MSE (Pred 2)", "CCE (Pred 1)", "CCE (Pred 2)"]
loss_values = [mse_loss_1, mse_loss_2, cce_loss_1, cce_loss_2]

plt.figure(figsize=(8,5))
plt.bar(labels, loss_values, color=['blue', 'lightblue', 'red', 'pink'])
plt.xlabel("Loss Type")
plt.ylabel("Loss Value")
plt.title("Comparison of MSE and CCE Losses")
plt.show()
•	A bar chart is created to visually compare the loss values for MSE and CCE.
 Output 
•	The script prints the loss values for both MSE and CCE for each set of predictions.
•	A bar chart visualizing the differences in loss values is displayed.
Key Learnings and Features
Loss Functions: Comparison of MSE and CCE in different prediction scenarios. TensorFlow Implementation: How to use TensorFlow to compute loss values. Visualization: Matplotlib is used to compare loss values effectively.
• Bar Chart Representation
•	Blue shades → MSE losses
•	Red shades → CCE losses
•	Shows how loss values change with different predictions
How to Run the Code
1.	Save the script as loss_comparison.py.
2.	Run the script using:
python loss_comparison.py
3.	The loss values will be printed, and a bar chart will be displayed.
Conclusion
This project illustrates the importance of choosing the right loss function for a given machine learning problem. MSE works better for regression, while CCE is crucial for classification tasks.

3.Train a Model with Different Optimizers
 Overview
This project compares the performance of two different optimization algorithms, Adam and SGD, in training a neural network on the Digits dataset (a smaller alternative to MNIST). It evaluates training and validation accuracy trends to determine which optimizer performs better.
Installation Instructions
1.	Install Python 
2.	Install required dependencies using pip:
pip install numpy matplotlib scikit-learn
3.	Ensure that your environment supports Jupyter Notebook or any Python IDE for running the script.
Usage Guide
Run the script using Python:
python train_model_optimizers.py
This will train two models, one with Adam and one with SGD, and visualize their accuracy comparison.
Code Explanation
1.	Load the Digits Dataset: The dataset is loaded from scikit-learn’s load_digits() function and normalized.
2.	Split into Train and Test Sets: Data is split into 80% training and 20% testing.
3.	Data Standardization: The training and test sets are standardized using StandardScaler.
4.	Train Models: Two MLP classifiers are trained using Adam and SGD optimizers with one hidden layer of 128 neurons.
5.	Evaluate Accuracy: Training and validation accuracy are computed for both models.
6.	Plot Results: A bar chart visualizing training and validation accuracy of both optimizers is generated.
Output
•	Training and validation accuracy values printed in the console.
•	A bar chart comparing the accuracy of models trained with Adam and SGD.
Key Learnings or Features
•	Adam tends to converge faster and perform better on complex datasets.
•	SGD may require careful tuning of the learning rate for optimal performance.
•	Data standardization improves the training process.
•	Visualization helps compare the effectiveness of different optimizers.
This project provides insight into optimizer selection for machine learning tasks and demonstrates key differences between Adam and SGD.

4.	Train a Neural Network and Log to TensorBoard
Overview
This project demonstrates how to train a neural network using the MNIST-like Digits dataset and log its performance metrics to TensorBoard. The model is a simple Multi-Layer Perceptron (MLP) classifier that learns to recognize handwritten digits. The training accuracy and test accuracy are visualized using a bar chart.
Installation Instructions
To run this project, ensure you have the following dependencies installed:
pip install numpy matplotlib scikit-learn
Usage Guide
Run the following command to execute the script:
python script.py

Code Explanation
1.	Load the Dataset:
o	The MNIST-like Digits dataset is loaded from sklearn.datasets.load_digits().
o	The pixel values are normalized by dividing by 16.0.
2.	Preprocessing:
o	The dataset is split into training (80%) and test (20%) sets.
o	Features are standardized using StandardScaler().
3.	Model Definition & Training:
o	An MLP classifier is defined with one hidden layer of 128 neurons, ReLU activation, and the Adam optimizer.
o	The model is trained for 5 epochs.
4.	Evaluation:
o	The training and test accuracy are computed using accuracy_score().
5.	Visualization:
o	A bar chart displays training and test accuracy for comparison.
Expected Output
•	The model prints training and test accuracy to the console.
•	A bar chart visualizing accuracy trends is generated.
Key Learnings or Features
•	Implementing a simple neural network for digit classification.
•	Standardizing input data for better model performance.
•	Using bar charts to compare training and test accuracy.
•	Logging model performance using TensorBoard for deeper insights.
