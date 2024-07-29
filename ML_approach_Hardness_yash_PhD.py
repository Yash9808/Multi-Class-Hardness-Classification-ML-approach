import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix,  classification_report, accuracy_score
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the data from the CSV file
data32a = pd.read_csv('VPF_spikes.csv')
data32=data32a[0:800]
# Split the data into features and target variable
X = data32[['F-volt', 'V-volt', 'P-volt', 'FA1','SA1','FA2','SA2' ]].values
y = data32['Object'].values

# Convert string labels to numeric values
label_mapping = {'H': 0, 'S': 1,'F': 2}
y = np.array([label_mapping[label] for label in y])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Logistic Regression model
lr = LogisticRegression(multi_class='multinomial', solver='lbfgs')
lr.fit(X_train, y_train)

# Train a Support Vector Classification model 'kernel': ['linear', 'rbf']
svc = SVC(kernel='linear', decision_function_shape='ovr')
svc.fit(X_train, y_train)

# Train a Support Vector Classification model 'kernel': ['linear', 'rbf']
svc1 = SVC(kernel='rbf', decision_function_shape='ovr')
svc1.fit(X_train, y_train)

# Train a Decision Tree Classifier model
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

# Train a Random Forest Classifier model'max_features': ['auto', 'sqrt', 'log2']
rfc = RandomForestClassifier(max_features='auto')
rfc.fit(X_train, y_train)

# Initialize and train k-Nearest Neighbors (k-NN)
knn_classifier = KNeighborsClassifier(n_neighbors=5)  # You can adjust k as needed
knn_classifier.fit(X_train, y_train)


# Initialize and train Artificial Neural Network (ANN)
# You can adjust the parameters for the MLPClassifier as needed
ann_classifier = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
ann_classifier.fit(X_train, y_train)

# Predict the target variable for the testing set
lr_pred = lr.predict(X_test)
svc_pred = svc.predict(X_test)
svc_pred1 = svc1.predict(X_test)
dtc_pred = dtc.predict(X_test)
rfc_pred = rfc.predict(X_test)
y_pred_knn = knn_classifier.predict(X_test)
y_pred_ann = ann_classifier.predict(X_test)

# Create confusion matrices for each classifier
confusion_lr = confusion_matrix(y_test, lr_pred)
confusion_svc = confusion_matrix(y_test, svc_pred)
confusion_dtc = confusion_matrix(y_test, dtc_pred)
confusion_rfc = confusion_matrix(y_test, rfc_pred)

# Create classification reports for each algorithm
report_lr = classification_report(y_test, lr_pred)
report_svc = classification_report(y_test, svc_pred)
report_dtc = classification_report(y_test, dtc_pred)
report_rfc = classification_report(y_test, rfc_pred)

'''print("Logistic Regression Report:")
print(report_lr)
print("\nSupport Vector Classifier Report:")
print(report_svc)
print("\nDecision Tree Classifier Report:")
print(report_dtc)
print("\nRandom Forest Classifier Report:")
print(report_rfc)'''

# Calculate accuracy for each algorithm
accuracy_lr = np.mean(lr_pred == y_test) * 100
accuracy_svc = np.mean(svc_pred == y_test) * 100
accuracy_svc1 = np.mean(svc_pred1 == y_test) * 100
accuracy_dtc = np.mean(dtc_pred == y_test) * 100
accuracy_rfc = np.mean(rfc_pred == y_test) * 100
accuracy_knn = accuracy_score(y_test, y_pred_knn)*100
accuracy_ann = accuracy_score(y_test, y_pred_ann)*100

# Print accuracy
print(f"Logistic Regression Accuracy: {accuracy_lr:.2f}%")
print(f"Support Vector Classifier Accuracy- Linear: {accuracy_svc:.2f}%")
print(f"Decision Tree Classifier Accuracy: {accuracy_dtc:.2f}%")
print(f"Random Forest Classifier Accuracy: {accuracy_rfc:.2f}%")
print(f'k-NN Accuracy: {accuracy_knn:.2f}%')
print(f'ANN Accuracy-MLP: {accuracy_ann:.2f}%')
print(f"Support Vector Classifier Accuracy- rbf: {accuracy_svc1:.2f}%")
