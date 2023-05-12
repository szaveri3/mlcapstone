# Import sk learn libraries for logistic regression, decision tree and neural network
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc

# Import ensemble for majority voting scheme
from sklearn.ensemble import VotingClassifier

# Import numpy for array manipulation
import numpy as np
import pandas as pd

# Import pickle to load the dataframe
import pickle

# Import matplotlib for plotting
import matplotlib.pyplot as plt

# Load the dataframe in a function
def load_df():
    with open('data/master_df.pickle', 'rb') as f:
        master_df = pickle.load(f)
    return master_df

# Functions for each model
def logistic_regression(X_train, y_train, X_test, y_test):
    # Create a logistic regression model with parameters 
    log_reg = LogisticRegression(C=0.01, random_state=42)

    # Train the model
    log_reg.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = log_reg.predict(X_test)

    # Return the predictions
    return y_pred

def decision_tree(X_train, y_train, X_test, y_test):
    # Create a decision tree model with parameters
    dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=0.16, random_state=42)

    # Train the model
    dt.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = dt.predict(X_test)

    # Return the predictions
    return y_pred

def neural_network(X_train, y_train, X_test, y_test):
    # Create a neural network model with parameters
    nn = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=400, random_state=42)

    # Train the model
    nn.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = nn.predict(X_test)

    # Return the predictions
    return y_pred

def majority_classifier(X_train, y_train, X_test, y_test):
    # Create a logistic regression model with parameters
    log_reg = LogisticRegression(C=0.01, random_state=42)

    # Create a decision tree model with parameters
    dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=0.16, random_state=42)

    # Create a neural network model with parameters
    nn = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=400, random_state=42)

    # Create a voting classifier
    voting_classifier = VotingClassifier(estimators=[('lr', log_reg), ('dt', dt), ('nn', nn)], voting='hard')

    # Train the model
    voting_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = voting_classifier.predict(X_test)

    # Return the predictions
    return y_pred

# Function to calculate the metrics
def calculate_metrics(y_test, y_pred):
    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate the precision
    precision = precision_score(y_test, y_pred)

    # Calculate the recall
    recall = recall_score(y_test, y_pred)

    # Return the metrics
    return accuracy, precision, recall


