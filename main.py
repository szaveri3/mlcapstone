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

    print('Master dataframe:')
    print(master_df.head())

    # Create the features and labels, label is atherosclerosis
    X = master_df[['Hypertension', 'Hypercholesterolemia', 'Male', 'Female', 'Age <40', 'Age 40-59', 'Age 60-79', 'Age 80+']]
    y = master_df['Atherosclerosis']

    # Split the data into training and testing sets, 0.8 and 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Run logistic regression with different values of C
    C = [0.01, 0.1, 1, 10, 100]
    log_accuracy = []
    log_precision = []
    log_recall = []
    for c in C:
        y_pred = logistic_regression(c, X_train, y_train, X_test, y_test)
        accuracy, precision, recall = calculate_metrics(y_test, y_pred)
        log_accuracy.append(accuracy)
        log_precision.append(precision)
        log_recall.append(recall)

        print('Logistic regression with C = {}:'.format(c))
        print('Accuracy: {}'.format(accuracy))


    # Run decision tree with different large values of max_depth incrementing by 5
    max_depth = [5, 10, 15, 20, 25, 30, 35, 40]
    dt_accuracy = []
    dt_precision = []
    dt_recall = []
    for depth in max_depth:
        y_pred = decision_tree(depth, X_train, y_train, X_test, y_test)
        accuracy, precision, recall = calculate_metrics(y_test, y_pred)
        dt_accuracy.append(accuracy)
        dt_precision.append(precision)
        dt_recall.append(recall)

        print('Decision tree with max_depth = {}:'.format(depth))
        print('Accuracy: {}'.format(accuracy))

    # Plot the accuracy of decision tree with different values of max_depth, fix scale
    plt.figure()
    plt.plot(max_depth, dt_accuracy)
    plt.xlabel('max_depth')
    plt.ylabel('Accuracy')
    plt.title('Decision Tree Accuracy vs max_depth')
    plt.show()

    # Plot the accuracy of logistic regression with different values of C, fix scale
    plt.figure()
    plt.plot(C, log_accuracy)
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.title('Logistic Regression Accuracy vs C')
    plt.show()
    


if __name__ == '__main__':
    main()





