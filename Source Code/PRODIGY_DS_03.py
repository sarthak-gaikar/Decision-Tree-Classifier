import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import numpy as np

# Loading the dataset
bank_data = pd.read_csv('bank-additional-full.csv', sep=';', quotechar='"')

# Data preprocessing
# Dropping rows with missing values
bank_data.dropna(inplace=True)

# Encode categorical variables
label_encoders = {}
for column in bank_data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    bank_data[column] = label_encoders[column].fit_transform(bank_data[column])

# Split dataset into features and target variable
X = bank_data.drop('y', axis=1)
y = bank_data['y']

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build decision tree classifier
clf = DecisionTreeClassifier(class_weight={0: 1, 1: 2})

# Train the model using the training sets
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate model performance
print("Decision Tree Classifier:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Visualize the decision tree
plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=['No', 'Yes'], max_depth=3)
plt.show()

import seaborn as sns
from sklearn.metrics import confusion_matrix

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Define class labels
classes = ['No Purchase', 'Purchase']

# Plot confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

plt.figure(figsize=(8, 6))
bar_width = 0.35
index = np.arange(len(classes))

# Plot bars for true labels
plt.bar(index, cm.sum(axis=1), bar_width, label='Actual', color='b', edgecolor='white')

# Plot bars for predicted labels
plt.bar(index + bar_width, cm.sum(axis=0), bar_width, label='Predicted', color='r', edgecolor='white')

plt.xlabel('Class')
plt.ylabel('Number of Instances')
plt.title('Actual vs. Predicted Class')
plt.xticks(index + bar_width / 2, classes)
plt.legend()

plt.tight_layout()
plt.show()

# Compute number of actual positive instances for each class
actual_positives = np.sum((y_test == 1), axis=0)

# Compute recall for each class based on actual instances
recall_actual_instances = cm.diagonal() / actual_positives

# Plot recall values based on actual instances
plt.figure(figsize=(8, 6))
plt.bar(classes, recall_actual_instances, color='c')
plt.xlabel('Class')
plt.ylabel('Recall')
plt.title('Recall Based on Actual Instances for Each Class')
plt.ylim(0, 1)
plt.show()