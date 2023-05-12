# Import other files
from models import load_df, logistic_regression, decision_tree, neural_network, majority_classifier, calculate_metrics
from preprocess import preprocessAdmissions, preprocessDiagnoses, preprocessPatients
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def main():
    # Load and preprocess the data
    preprocessAdmissions()
    preprocessDiagnoses()
    preprocessPatients()

    # Load the dataframe
    master_df = load_df()

    # Create the features and labels, label is atherosclerosis
    X = master_df[['Hypertension', 'Hypercholesterolemia', 'Male', 'Female', 'Age <40', 'Age 40-59', 'Age 60-79', 'Age 80+']]
    y = master_df['Atherosclerosis']

    # Split the data into training and testing sets, 0.8 and 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create a list of models
    models = [logistic_regression, decision_tree, neural_network, majority_classifier]

    # Create a list of model names
    model_names = ['Logistic Regression', 'Decision Tree', 'Neural Network', 'Majority Classifier']

    # Create a list of model predictions
    model_predictions = []

    # Create a list of model accuracies
    model_accuracies = []

    # Create a list of model precisions
    model_precisions = []

    # Create a list of model recalls
    model_recalls = []

    # Run each model
    for model in models:
        # Make predictions
        y_pred = model(X_train, y_train, X_test, y_test)

        # Calculate metrics
        accuracy, precision, recall = calculate_metrics(y_test, y_pred)

        # Append the predictions, accuracies, precisions, and recalls to their lists
        model_predictions.append(y_pred)
        model_accuracies.append(accuracy)
        model_precisions.append(precision)
        model_recalls.append(recall)

    # Create a dataframe of the metrics
    metrics_df = pd.DataFrame({'Model': model_names, 'Accuracy': model_accuracies, 'Precision': model_precisions, 'Recall': model_recalls})

    # Print the dataframe
    print(metrics_df)

    


if __name__ == '__main__':
    main()





