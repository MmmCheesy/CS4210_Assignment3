#-------------------------------------------------------------------------
# AUTHOR: Alexander Eckert
# FILENAME: svm.py
# SPECIFICATION: Using svm to determine handwritten letters.
# FOR: CS 4210- Assignment #3
# TIME SPENT: 47 hrs, 13 min
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn import svm
import numpy as np
import pandas as pd

# Define the hyperparameter values
c_values = [1, 5, 10, 100]
degree_values = [1, 2, 3]
kernel_values = ["linear", "poly", "rbf"]
decision_function_shape_values = ["ovo", "ovr"]

df = pd.read_csv('optdigits.tra', sep=',', header=None)
X_training = np.array(df.values)[:, :64]
y_training = np.array(df.values)[:, -1]

df = pd.read_csv('optdigits.tes', sep=',', header=None)
X_test = np.array(df.values)[:, :64]
y_test = np.array(df.values)[:, -1]

highest_accuracy = 0
best_parameters = None

for c in c_values:
    for degree in degree_values:
        for kernel in kernel_values:
            for decision_function_shape in decision_function_shape_values:
                # Create an SVM classifier with the current hyperparameters
                clf = svm.SVC(C=c, degree=degree, kernel=kernel, decision_function_shape=decision_function_shape)

                # Fit SVM to the training data
                clf.fit(X_training, y_training)

                # Make predictions on the test data and compute accuracy
                correct_predictions = 0
                total_predictions = len(X_test)

                for x_testSample, y_testSample in zip(X_test, y_test):
                    prediction = clf.predict([x_testSample])
                    if prediction[0] == y_testSample:
                        correct_predictions += 1

                accuracy = correct_predictions / total_predictions

                # Check if the accuracy is higher than the previous highest accuracy
                if accuracy > highest_accuracy:
                    highest_accuracy = accuracy
                    best_parameters = (c, degree, kernel, decision_function_shape)
                    print(f"Highest SVM accuracy so far: {highest_accuracy:.2f}, Parameters: c={c}, degree={degree}, kernel={kernel}, decision_function_shape='{decision_function_shape}'")

print(f"Best SVM parameters: c={best_parameters[0]}, degree={best_parameters[1]}, kernel={best_parameters[2]}, decision_function_shape='{best_parameters[3]}'")
print(f"Best SVM accuracy: {highest_accuracy:.2f}")





