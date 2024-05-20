# Importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from itertools import combinations

# Step 1: Load the dataset using pandas read_csv method.
heart_disease = pd.read_csv("heart_disease1.csv")
print("Dataset: ")
print(heart_disease)

# Step 2: Pre-process the data by handling missing values.
# handling missing values
heart_disease = heart_disease.fillna(0)

# Step 3: Apply the WARM algorithm to extract rules from the dataset.
def WARM(transactions, min_sup, min_conf, max_len):
    itemsets = {}
    transactions = [set(transaction) for transaction in transactions]
    items = set(item for transaction in transactions for item in transaction)
    for i in range(1, max_len+1):
        itemsets[i] = []
        for itemset in combinations(items, i):
            count = 0
            for transaction in transactions:
                if set(itemset).issubset(transaction):
                    count += 1
            support = count / len(transactions)
            if support >= min_sup:
                itemsets[i].append((tuple(sorted(itemset)), support))
    rules = []
    for i in range(1, max_len):
        for itemset, support in itemsets[i]:
            for j in range(i+1, max_len+1):
                for itemset2, support2 in itemsets[j]:
                    if set(itemset).issubset(itemset2):
                        conf = support2 / support
                        if conf >= min_conf:
                            rules.append((itemset, set(itemset2) - set(itemset), conf))
    return rules

transactions = heart_disease.values.tolist()
rules = WARM(transactions, 0.2, 0.5, 3)
print("\nWARM Rules:")
print(rules)

# Step 4: Split the dataset into training and testing sets.
X = heart_disease.iloc[:, :-1]
y = heart_disease.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nX_train:")
print(X_train)

print("\ny_train:")
print(y_train)

print("\nX_test:")
print(X_test)

print("\ny_test:")
print(y_test)

# Step 5: Apply the Decision Tree algorithm to the training set, predict the 
# target variable using the trained model, and evaluate the model's performance 
# using various metrics - accuracy_score, precision_score, recall_score, f1_score, 
# confusion_matrix.
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Predicting the target variable using the DT model
y_pred = dt.predict(X_test)
print("\nDecision Tree Predicted Results:")
print(y_pred)

# Evaluating the DT model's performance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# calculate evaluation metrics
dt_accuracy = accuracy_score(y_test, y_pred)
dt_precision = precision_score(y_test, y_pred, average='weighted')
dt_recall = recall_score(y_test, y_pred, average='weighted')
dt_f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

# print evaluation metrics
print(f'DT Accuracy: {dt_accuracy:.2f}')
print(f'DT Precision: {dt_precision:.2f}')
print(f'DT Recall: {dt_recall:.2f}')
print(f'DT F1 Score: {dt_f1:.2f}')
print('DT Confusion Matrix:')
print(conf_matrix)

# Step 6: Apply the Naive Bayes algorithm to the training set, predict the 
# target variable using the trained model, and evaluate the model's performance 
# using various metrics - accuracy_score, precision_score, recall_score, f1_score, 
# confusion_matrix.

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)

# Predicting the target variable using the Naive Bayes model
y_pred = nb.predict(X_test)
print("\nNaive Bayes Predicted Results:")
print(y_pred)

# Evaluating the Naive Bayes model's performance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# calculate evaluation metrics
nb_accuracy = accuracy_score(y_test, y_pred)
nb_precision = precision_score(y_test, y_pred, average='weighted')
nb_recall = recall_score(y_test, y_pred, average='weighted')
nb_f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

# print evaluation metrics
print(f'Naive Bayes Accuracy: {nb_accuracy:.2f}')
print(f'Naive Bayes Precision: {nb_precision:.2f}')
print(f'Naive Bayes Recall: {nb_recall:.2f}')
print(f'Naive Bayes F1 Score: {nb_f1:.2f}')
print('Naive Bayes Confusion Matrix:')
print(conf_matrix)

