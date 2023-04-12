#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os, glob
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.decomposition import PCA
import shap
# Split the data
from sklearn.model_selection import train_test_split
# Define the model
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


# In[2]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# # Modeling 

# # Applying Random Forest on all scenarios

# In[ ]:


rf = RandomForestClassifier(n_estimators=50, random_state=42)
filename = f"/Users/kritika/Documents/data2/1features.txt"
#print('-'*100)
df = pd.read_csv(filename,sep = ',', header=None)
y = df.iloc[:, 0] #labels are in the first column
X = df.iloc[:, 1:] #features are in the rest of the columns
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.25)
    
    
# Train the random forest classifier on the training data
rf.fit(X_train, y_train)
    
# Predict the labels for the test data
y_pred = rf.predict(X_test)
    
# Calculate the accuracy of the predictions
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Calculate the AUC score
auc_score = roc_auc_score(y_test, y_pred)

# Calculate the mean AUC score
mean_auc_score = np.mean(auc_scores)

# Print the mean AUC score
print('Mean AUC Score:', mean_auc_score)


# In[ ]:


rf = RandomForestClassifier(n_estimators=50, random_state=42)
filename = f"/Users/kritika/Documents/data2/2features.txt"
#print('-'*100)
df = pd.read_csv(filename,sep = ',', header=None)
y = df.iloc[:, 0] #labels are in the first column
X = df.iloc[:, 1:] #features are in the rest of the columns
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.25)
    
    
# Train the random forest classifier on the training data
rf.fit(X_train, y_train)
    
# Predict the labels for the test data
y_pred = rf.predict(X_test)
    
# Calculate the accuracy of the predictions
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Calculate the AUC score
auc_score = roc_auc_score(y_test, y_pred)

# Calculate the mean AUC score
mean_auc_score = np.mean(auc_scores)

# Print the mean AUC score
print('Mean AUC Score:', mean_auc_score)


# In[ ]:


rf = RandomForestClassifier(n_estimators=50, random_state=42)
filename = f"/Users/kritika/Documents/data2/3features.txt"
#print('-'*100)
df = pd.read_csv(filename,sep = ',', header=None)
y = df.iloc[:, 0] #labels are in the first column
X = df.iloc[:, 1:] #features are in the rest of the columns
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.25)
    
    
# Train the random forest classifier on the training data
rf.fit(X_train, y_train)
    
# Predict the labels for the test data
y_pred = rf.predict(X_test)
    
# Calculate the accuracy of the predictions
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Calculate the AUC score
auc_score = roc_auc_score(y_test, y_pred)

# Calculate the mean AUC score
mean_auc_score = np.mean(auc_scores)

# Print the mean AUC score
print('Mean AUC Score:', mean_auc_score)


# In[ ]:


rf = RandomForestClassifier(n_estimators=50, random_state=42)
filename = f"/Users/kritika/Documents/data2/4features.txt"
#print('-'*100)
df = pd.read_csv(filename,sep = ',', header=None)
y = df.iloc[:, 0] #labels are in the first column
X = df.iloc[:, 1:] #features are in the rest of the columns
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.25)
    
    
# Train the random forest classifier on the training data
rf.fit(X_train, y_train)
    
# Predict the labels for the test data
y_pred = rf.predict(X_test)
    
# Calculate the accuracy of the predictions
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Calculate the AUC score
auc_score = roc_auc_score(y_test, y_pred)

# Calculate the mean AUC score
mean_auc_score = np.mean(auc_scores)

# Print the mean AUC score
print('Mean AUC Score:', mean_auc_score)


# In[ ]:


rf = RandomForestClassifier(n_estimators=50, random_state=42)
filename = f"/Users/kritika/Documents/data2/5features.txt"
#print('-'*100)
df = pd.read_csv(filename,sep = ',', header=None)
y = df.iloc[:, 0] #labels are in the first column
X = df.iloc[:, 1:] #features are in the rest of the columns
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.25)
    
    
# Train the random forest classifier on the training data
rf.fit(X_train, y_train)
    
# Predict the labels for the test data
y_pred = rf.predict(X_test)
    
# Calculate the accuracy of the predictions
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Calculate the AUC score
auc_score = roc_auc_score(y_test, y_pred)

# Calculate the mean AUC score
mean_auc_score = np.mean(auc_scores)

# Print the mean AUC score
print('Mean AUC Score:', mean_auc_score)


# In[ ]:


rf = RandomForestClassifier(n_estimators=50, random_state=42)
filename = f"/Users/kritika/Documents/data2/6features.txt"
#print('-'*100)
df = pd.read_csv(filename,sep = ',', header=None)
y = df.iloc[:, 0] #labels are in the first column
X = df.iloc[:, 1:] #features are in the rest of the columns
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.25)
    
    
# Train the random forest classifier on the training data
rf.fit(X_train, y_train)
    
# Predict the labels for the test data
y_pred = rf.predict(X_test)
    
# Calculate the accuracy of the predictions
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Calculate the AUC score
auc_score = roc_auc_score(y_test, y_pred)

# Calculate the mean AUC score
mean_auc_score = np.mean(auc_scores)

# Print the mean AUC score
print('Mean AUC Score:', mean_auc_score)


# In[1]:


rf = RandomForestClassifier(n_estimators=50, random_state=42)
filename = f"/Users/kritika/Documents/data2/7features.txt"
#print('-'*100)
df = pd.read_csv(filename,sep = ',', header=None)
y = df.iloc[:, 0] #labels are in the first column
X = df.iloc[:, 1:] #features are in the rest of the columns
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.25)
    
    
# Train the random forest classifier on the training data
rf.fit(X_train, y_train)
    
# Predict the labels for the test data
y_pred = rf.predict(X_test)
    
# Calculate the accuracy of the predictions
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Calculate the AUC score
auc_score = roc_auc_score(y_test, y_pred)

# Calculate the mean AUC score
mean_auc_score = np.mean(auc_scores)

# Print the mean AUC score
print('Mean AUC Score:', mean_auc_score)


# In[ ]:


rf = RandomForestClassifier(n_estimators=50, random_state=42)
filename = f"/Users/kritika/Documents/data2/8features.txt"
#print('-'*100)
df = pd.read_csv(filename,sep = ',', header=None)
y = df.iloc[:, 0] #labels are in the first column
X = df.iloc[:, 1:] #features are in the rest of the columns
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.25)
    
    
# Train the random forest classifier on the training data
rf.fit(X_train, y_train)
    
# Predict the labels for the test data
y_pred = rf.predict(X_test)
    
# Calculate the accuracy of the predictions
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Calculate the AUC score
auc_score = roc_auc_score(y_test, y_pred)

# Calculate the mean AUC score
mean_auc_score = np.mean(auc_scores)

# Print the mean AUC score
print('Mean AUC Score:', mean_auc_score)


# In[ ]:


rf = RandomForestClassifier(n_estimators=50, random_state=42)
filename = f"/Users/kritika/Documents/data2/9features.txt"
#print('-'*100)
df = pd.read_csv(filename,sep = ',', header=None)
y = df.iloc[:, 0] #labels are in the first column
X = df.iloc[:, 1:] #features are in the rest of the columns
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.25)
    
    
# Train the random forest classifier on the training data
rf.fit(X_train, y_train)
    
# Predict the labels for the test data
y_pred = rf.predict(X_test)
    
# Calculate the accuracy of the predictions
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Calculate the AUC score
auc_score = roc_auc_score(y_test, y_pred)

# Calculate the mean AUC score
mean_auc_score = np.mean(auc_scores)

# Print the mean AUC score
print('Mean AUC Score:', mean_auc_score)


# In[ ]:


rf = RandomForestClassifier(n_estimators=50, random_state=42)
filename = f"/Users/kritika/Documents/data2/10features.txt"
#print('-'*100)
df = pd.read_csv(filename,sep = ',', header=None)
y = df.iloc[:, 0] #labels are in the first column
X = df.iloc[:, 1:] #features are in the rest of the columns
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.25)
    
    
# Train the random forest classifier on the training data
rf.fit(X_train, y_train)
    
# Predict the labels for the test data
y_pred = rf.predict(X_test)
    
# Calculate the accuracy of the predictions
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Calculate the AUC score
auc_score = roc_auc_score(y_test, y_pred)

# Calculate the mean AUC score
mean_auc_score = np.mean(auc_scores)

# Print the mean AUC score
print('Mean AUC Score:', mean_auc_score)


# In[ ]:


rf = RandomForestClassifier(n_estimators=50, random_state=42)
filename = f"/Users/kritika/Documents/data2/11features.txt"
#print('-'*100)
df = pd.read_csv(filename,sep = ',', header=None)
y = df.iloc[:, 0] #labels are in the first column
X = df.iloc[:, 1:] #features are in the rest of the columns
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.25)
    
    
# Train the random forest classifier on the training data
rf.fit(X_train, y_train)
    
# Predict the labels for the test data
y_pred = rf.predict(X_test)
    
# Calculate the accuracy of the predictions
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Calculate the AUC score
auc_score = roc_auc_score(y_test, y_pred)

# Calculate the mean AUC score
mean_auc_score = np.mean(auc_scores)

# Print the mean AUC score
print('Mean AUC Score:', mean_auc_score)


# In[ ]:


rf = RandomForestClassifier(n_estimators=50, random_state=42)
filename = f"/Users/kritika/Documents/data2/12features.txt"
#print('-'*100)
df = pd.read_csv(filename,sep = ',', header=None)
y = df.iloc[:, 0] #labels are in the first column
X = df.iloc[:, 1:] #features are in the rest of the columns
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.25)
    
    
# Train the random forest classifier on the training data
rf.fit(X_train, y_train)
    
# Predict the labels for the test data
y_pred = rf.predict(X_test)
    
# Calculate the accuracy of the predictions
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Calculate the AUC score
auc_score = roc_auc_score(y_test, y_pred)

# Calculate the mean AUC score
mean_auc_score = np.mean(auc_scores)

# Print the mean AUC score
print('Mean AUC Score:', mean_auc_score)


# In[ ]:


# rf = RandomForestClassifier(n_estimators=50, random_state=42)
filename = f"/Users/kritika/Documents/data2/13features.txt"
#print('-'*100)
df = pd.read_csv(filename,sep = ',', header=None)
y = df.iloc[:, 0] #labels are in the first column
X = df.iloc[:, 1:] #features are in the rest of the columns
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.25)
    
    
# Train the random forest classifier on the training data
rf.fit(X_train, y_train)
    
# Predict the labels for the test data
y_pred = rf.predict(X_test)
    
# Calculate the accuracy of the predictions
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Calculate the AUC score
auc_score = roc_auc_score(y_test, y_pred)

# Calculate the mean AUC score
mean_auc_score = np.mean(auc_scores)

# Print the mean AUC score
print('Mean AUC Score:', mean_auc_score)


# # Applying KNN on all scenarios
# 

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
filename = f"/Users/kritika/Documents/data2/1features.txt"
#print('-'*100)
df = pd.read_csv(filename,sep = ',', header=None)
y = df.iloc[:, 0] #labels are in the first column
X = df.iloc[:, 1:] #features are in the rest of the columns
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.25)
    
# Instantiate KNN classifier with k=5
knn = KNeighborsClassifier(n_neighbors=5)

# Train KNN model on training data
knn.fit(X_train, y_train)

# Predict classes of test data using trained KNN model
y_pred = knn.predict(X_test)

# Evaluate accuracy of KNN model on test data
accuracy = accuracy_score(y_test, y_pred)

# Calculate AUC score of KNN model on test data
auc = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'AUC score: {auc}')


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
filename = f"/Users/kritika/Documents/data2/2features.txt"
#print('-'*100)
df = pd.read_csv(filename,sep = ',', header=None)
y = df.iloc[:, 0] #labels are in the first column
X = df.iloc[:, 1:] #features are in the rest of the columns
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.25)
    
# Instantiate KNN classifier with k=5
knn = KNeighborsClassifier(n_neighbors=5)

# Train KNN model on training data
knn.fit(X_train, y_train)

# Predict classes of test data using trained KNN model
y_pred = knn.predict(X_test)

# Evaluate accuracy of KNN model on test data
accuracy = accuracy_score(y_test, y_pred)

# Calculate AUC score of KNN model on test data
auc = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'AUC score: {auc}')


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
filename = f"/Users/kritika/Documents/data2/3features.txt"
#print('-'*100)
df = pd.read_csv(filename,sep = ',', header=None)
y = df.iloc[:, 0] #labels are in the first column
X = df.iloc[:, 1:] #features are in the rest of the columns
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.25)
    
# Instantiate KNN classifier with k=5
knn = KNeighborsClassifier(n_neighbors=5)

# Train KNN model on training data
knn.fit(X_train, y_train)

# Predict classes of test data using trained KNN model
y_pred = knn.predict(X_test)

# Evaluate accuracy of KNN model on test data
accuracy = accuracy_score(y_test, y_pred)

# Calculate AUC score of KNN model on test data
auc = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'AUC score: {auc}')


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
filename = f"/Users/kritika/Documents/data2/4features.txt"
#print('-'*100)
df = pd.read_csv(filename,sep = ',', header=None)
y = df.iloc[:, 0] #labels are in the first column
X = df.iloc[:, 1:] #features are in the rest of the columns
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.25)
    
# Instantiate KNN classifier with k=5
knn = KNeighborsClassifier(n_neighbors=5)

# Train KNN model on training data
knn.fit(X_train, y_train)

# Predict classes of test data using trained KNN model
y_pred = knn.predict(X_test)

# Evaluate accuracy of KNN model on test data
accuracy = accuracy_score(y_test, y_pred)

# Calculate AUC score of KNN model on test data
auc = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'AUC score: {auc}')


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
filename = f"/Users/kritika/Documents/data2/5features.txt"
#print('-'*100)
df = pd.read_csv(filename,sep = ',', header=None)
y = df.iloc[:, 0] #labels are in the first column
X = df.iloc[:, 1:] #features are in the rest of the columns
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.25)
    
# Instantiate KNN classifier with k=5
knn = KNeighborsClassifier(n_neighbors=5)

# Train KNN model on training data
knn.fit(X_train, y_train)

# Predict classes of test data using trained KNN model
y_pred = knn.predict(X_test)

# Evaluate accuracy of KNN model on test data
accuracy = accuracy_score(y_test, y_pred)

# Calculate AUC score of KNN model on test data
auc = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'AUC score: {auc}')


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
filename = f"/Users/kritika/Documents/data2/6features.txt"
#print('-'*100)
df = pd.read_csv(filename,sep = ',', header=None)
y = df.iloc[:, 0] #labels are in the first column
X = df.iloc[:, 1:] #features are in the rest of the columns
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.25)
    
# Instantiate KNN classifier with k=5
knn = KNeighborsClassifier(n_neighbors=5)

# Train KNN model on training data
knn.fit(X_train, y_train)

# Predict classes of test data using trained KNN model
y_pred = knn.predict(X_test)

# Evaluate accuracy of KNN model on test data
accuracy = accuracy_score(y_test, y_pred)

# Calculate AUC score of KNN model on test data
auc = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'AUC score: {auc}')


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
filename = f"/Users/kritika/Documents/data2/7features.txt"
#print('-'*100)
df = pd.read_csv(filename,sep = ',', header=None)
y = df.iloc[:, 0] #labels are in the first column
X = df.iloc[:, 1:] #features are in the rest of the columns
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.25)
    
# Instantiate KNN classifier with k=5
knn = KNeighborsClassifier(n_neighbors=5)

# Train KNN model on training data
knn.fit(X_train, y_train)

# Predict classes of test data using trained KNN model
y_pred = knn.predict(X_test)

# Evaluate accuracy of KNN model on test data
accuracy = accuracy_score(y_test, y_pred)

# Calculate AUC score of KNN model on test data
auc = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'AUC score: {auc}')


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
filename = f"/Users/kritika/Documents/data2/8features.txt"
#print('-'*100)
df = pd.read_csv(filename,sep = ',', header=None)
y = df.iloc[:, 0] #labels are in the first column
X = df.iloc[:, 1:] #features are in the rest of the columns
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.25)
    
# Instantiate KNN classifier with k=5
knn = KNeighborsClassifier(n_neighbors=5)

# Train KNN model on training data
knn.fit(X_train, y_train)

# Predict classes of test data using trained KNN model
y_pred = knn.predict(X_test)

# Evaluate accuracy of KNN model on test data
accuracy = accuracy_score(y_test, y_pred)

# Calculate AUC score of KNN model on test data
auc = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'AUC score: {auc}')


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
filename = f"/Users/kritika/Documents/data2/9features.txt"
#print('-'*100)
df = pd.read_csv(filename,sep = ',', header=None)
y = df.iloc[:, 0] #labels are in the first column
X = df.iloc[:, 1:] #features are in the rest of the columns
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.25)
    
# Instantiate KNN classifier with k=5
knn = KNeighborsClassifier(n_neighbors=5)

# Train KNN model on training data
knn.fit(X_train, y_train)

# Predict classes of test data using trained KNN model
y_pred = knn.predict(X_test)

# Evaluate accuracy of KNN model on test data
accuracy = accuracy_score(y_test, y_pred)

# Calculate AUC score of KNN model on test data
auc = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'AUC score: {auc}')


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
filename = f"/Users/kritika/Documents/data2/10features.txt"
#print('-'*100)
df = pd.read_csv(filename,sep = ',', header=None)
y = df.iloc[:, 0] #labels are in the first column
X = df.iloc[:, 1:] #features are in the rest of the columns
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.25)
    
# Instantiate KNN classifier with k=5
knn = KNeighborsClassifier(n_neighbors=5)

# Train KNN model on training data
knn.fit(X_train, y_train)

# Predict classes of test data using trained KNN model
y_pred = knn.predict(X_test)

# Evaluate accuracy of KNN model on test data
accuracy = accuracy_score(y_test, y_pred)

# Calculate AUC score of KNN model on test data
auc = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'AUC score: {auc}')


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
filename = f"/Users/kritika/Documents/data2/11features.txt"
#print('-'*100)
df = pd.read_csv(filename,sep = ',', header=None)
y = df.iloc[:, 0] #labels are in the first column
X = df.iloc[:, 1:] #features are in the rest of the columns
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.25)
    
# Instantiate KNN classifier with k=5
knn = KNeighborsClassifier(n_neighbors=5)

# Train KNN model on training data
knn.fit(X_train, y_train)

# Predict classes of test data using trained KNN model
y_pred = knn.predict(X_test)

# Evaluate accuracy of KNN model on test data
accuracy = accuracy_score(y_test, y_pred)

# Calculate AUC score of KNN model on test data
auc = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'AUC score: {auc}')


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
filename = f"/Users/kritika/Documents/data2/12features.txt"
#print('-'*100)
df = pd.read_csv(filename,sep = ',', header=None)
y = df.iloc[:, 0] #labels are in the first column
X = df.iloc[:, 1:] #features are in the rest of the columns
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.25)
    
# Instantiate KNN classifier with k=5
knn = KNeighborsClassifier(n_neighbors=5)

# Train KNN model on training data
knn.fit(X_train, y_train)

# Predict classes of test data using trained KNN model
y_pred = knn.predict(X_test)

# Evaluate accuracy of KNN model on test data
accuracy = accuracy_score(y_test, y_pred)

# Calculate AUC score of KNN model on test data
auc = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'AUC score: {auc}')


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
filename = f"/Users/kritika/Documents/data2/13features.txt"
#print('-'*100)
df = pd.read_csv(filename,sep = ',', header=None)
y = df.iloc[:, 0] #labels are in the first column
X = df.iloc[:, 1:] #features are in the rest of the columns
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.25)
    
# Instantiate KNN classifier with k=5
knn = KNeighborsClassifier(n_neighbors=5)

# Train KNN model on training data
knn.fit(X_train, y_train)

# Predict classes of test data using trained KNN model
y_pred = knn.predict(X_test)

# Evaluate accuracy of KNN model on test data
accuracy = accuracy_score(y_test, y_pred)

# Calculate AUC score of KNN model on test data
auc = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'AUC score: {auc}')


# # Applying SVM on all scenarios
# 

# In[ ]:


from sklearn.svm import SVC
filename = f"/Users/kritika/Documents/data2/1features.txt"
#print('-'*100)
df = pd.read_csv(filename,sep = ',', header=None)
y = df.iloc[:, 0] #labels are in the first column
X = df.iloc[:, 1:] #features are in the rest of the columns
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.25)

# Instantiate SVM classifier with linear kernel
svm = SVC(kernel='linear')

# Train SVM model on training data
svm.fit(X_train, y_train)

# Predict classes of test data using trained SVM model
y_pred = svm.predict(X_test)

# Evaluate accuracy of SVM model on test data
accuracy = accuracy_score(y_test, y_pred)

# Calculate AUC score of SVM model on test data
auc = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'AUC score: {auc}')


# In[ ]:


from sklearn.svm import SVC
filename = f"/Users/kritika/Documents/data2/2features.txt"
#print('-'*100)
df = pd.read_csv(filename,sep = ',', header=None)
y = df.iloc[:, 0] #labels are in the first column
X = df.iloc[:, 1:] #features are in the rest of the columns
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.25)

# Instantiate SVM classifier with linear kernel
svm = SVC(kernel='linear')

# Train SVM model on training data
svm.fit(X_train, y_train)

# Predict classes of test data using trained SVM model
y_pred = svm.predict(X_test)

# Evaluate accuracy of SVM model on test data
accuracy = accuracy_score(y_test, y_pred)

# Calculate AUC score of SVM model on test data
auc = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'AUC score: {auc}')


# In[ ]:


from sklearn.svm import SVC
filename = f"/Users/kritika/Documents/data2/3features.txt"
#print('-'*100)
df = pd.read_csv(filename,sep = ',', header=None)
y = df.iloc[:, 0] #labels are in the first column
X = df.iloc[:, 1:] #features are in the rest of the columns
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.25)

# Instantiate SVM classifier with linear kernel
svm = SVC(kernel='linear')

# Train SVM model on training data
svm.fit(X_train, y_train)

# Predict classes of test data using trained SVM model
y_pred = svm.predict(X_test)

# Evaluate accuracy of SVM model on test data
accuracy = accuracy_score(y_test, y_pred)

# Calculate AUC score of SVM model on test data
auc = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'AUC score: {auc}')


# In[ ]:


from sklearn.svm import SVC
filename = f"/Users/kritika/Documents/data2/4features.txt"
#print('-'*100)
df = pd.read_csv(filename,sep = ',', header=None)
y = df.iloc[:, 0] #labels are in the first column
X = df.iloc[:, 1:] #features are in the rest of the columns
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.25)

# Instantiate SVM classifier with linear kernel
svm = SVC(kernel='linear')

# Train SVM model on training data
svm.fit(X_train, y_train)

# Predict classes of test data using trained SVM model
y_pred = svm.predict(X_test)

# Evaluate accuracy of SVM model on test data
accuracy = accuracy_score(y_test, y_pred)

# Calculate AUC score of SVM model on test data
auc = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'AUC score: {auc}')


# In[ ]:


from sklearn.svm import SVC
filename = f"/Users/kritika/Documents/data2/5features.txt"
#print('-'*100)
df = pd.read_csv(filename,sep = ',', header=None)
y = df.iloc[:, 0] #labels are in the first column
X = df.iloc[:, 1:] #features are in the rest of the columns
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.25)

# Instantiate SVM classifier with linear kernel
svm = SVC(kernel='linear')

# Train SVM model on training data
svm.fit(X_train, y_train)

# Predict classes of test data using trained SVM model
y_pred = svm.predict(X_test)

# Evaluate accuracy of SVM model on test data
accuracy = accuracy_score(y_test, y_pred)

# Calculate AUC score of SVM model on test data
auc = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'AUC score: {auc}')


# In[ ]:


from sklearn.svm import SVC
filename = f"/Users/kritika/Documents/data2/6features.txt"
#print('-'*100)
df = pd.read_csv(filename,sep = ',', header=None)
y = df.iloc[:, 0] #labels are in the first column
X = df.iloc[:, 1:] #features are in the rest of the columns
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.25)

# Instantiate SVM classifier with linear kernel
svm = SVC(kernel='linear')

# Train SVM model on training data
svm.fit(X_train, y_train)

# Predict classes of test data using trained SVM model
y_pred = svm.predict(X_test)

# Evaluate accuracy of SVM model on test data
accuracy = accuracy_score(y_test, y_pred)

# Calculate AUC score of SVM model on test data
auc = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'AUC score: {auc}')


# In[ ]:


from sklearn.svm import SVC
filename = f"/Users/kritika/Documents/data2/7features.txt"
#print('-'*100)
df = pd.read_csv(filename,sep = ',', header=None)
y = df.iloc[:, 0] #labels are in the first column
X = df.iloc[:, 1:] #features are in the rest of the columns
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.25)

# Instantiate SVM classifier with linear kernel
svm = SVC(kernel='linear')

# Train SVM model on training data
svm.fit(X_train, y_train)

# Predict classes of test data using trained SVM model
y_pred = svm.predict(X_test)

# Evaluate accuracy of SVM model on test data
accuracy = accuracy_score(y_test, y_pred)

# Calculate AUC score of SVM model on test data
auc = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'AUC score: {auc}')


# In[ ]:


from sklearn.svm import SVC
filename = f"/Users/kritika/Documents/data2/8features.txt"
#print('-'*100)
df = pd.read_csv(filename,sep = ',', header=None)
y = df.iloc[:, 0] #labels are in the first column
X = df.iloc[:, 1:] #features are in the rest of the columns
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.25)

# Instantiate SVM classifier with linear kernel
svm = SVC(kernel='linear')

# Train SVM model on training data
svm.fit(X_train, y_train)

# Predict classes of test data using trained SVM model
y_pred = svm.predict(X_test)

# Evaluate accuracy of SVM model on test data
accuracy = accuracy_score(y_test, y_pred)

# Calculate AUC score of SVM model on test data
auc = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'AUC score: {auc}')


# In[ ]:


from sklearn.svm import SVC
filename = f"/Users/kritika/Documents/data2/9features.txt"
#print('-'*100)
df = pd.read_csv(filename,sep = ',', header=None)
y = df.iloc[:, 0] #labels are in the first column
X = df.iloc[:, 1:] #features are in the rest of the columns
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.25)

# Instantiate SVM classifier with linear kernel
svm = SVC(kernel='linear')

# Train SVM model on training data
svm.fit(X_train, y_train)

# Predict classes of test data using trained SVM model
y_pred = svm.predict(X_test)

# Evaluate accuracy of SVM model on test data
accuracy = accuracy_score(y_test, y_pred)

# Calculate AUC score of SVM model on test data
auc = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'AUC score: {auc}')


# In[ ]:


from sklearn.svm import SVC
filename = f"/Users/kritika/Documents/data2/10features.txt"
#print('-'*100)
df = pd.read_csv(filename,sep = ',', header=None)
y = df.iloc[:, 0] #labels are in the first column
X = df.iloc[:, 1:] #features are in the rest of the columns
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.25)

# Instantiate SVM classifier with linear kernel
svm = SVC(kernel='linear')

# Train SVM model on training data
svm.fit(X_train, y_train)

# Predict classes of test data using trained SVM model
y_pred = svm.predict(X_test)

# Evaluate accuracy of SVM model on test data
accuracy = accuracy_score(y_test, y_pred)

# Calculate AUC score of SVM model on test data
auc = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'AUC score: {auc}')


# In[ ]:


from sklearn.svm import SVC
filename = f"/Users/kritika/Documents/data2/11features.txt"
#print('-'*100)
df = pd.read_csv(filename,sep = ',', header=None)
y = df.iloc[:, 0] #labels are in the first column
X = df.iloc[:, 1:] #features are in the rest of the columns
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.25)

# Instantiate SVM classifier with linear kernel
svm = SVC(kernel='linear')

# Train SVM model on training data
svm.fit(X_train, y_train)

# Predict classes of test data using trained SVM model
y_pred = svm.predict(X_test)

# Evaluate accuracy of SVM model on test data
accuracy = accuracy_score(y_test, y_pred)

# Calculate AUC score of SVM model on test data
auc = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'AUC score: {auc}')


# In[ ]:


from sklearn.svm import SVC
filename = f"/Users/kritika/Documents/data2/12features.txt"
#print('-'*100)
df = pd.read_csv(filename,sep = ',', header=None)
y = df.iloc[:, 0] #labels are in the first column
X = df.iloc[:, 1:] #features are in the rest of the columns
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.25)

# Instantiate SVM classifier with linear kernel
svm = SVC(kernel='linear')

# Train SVM model on training data
svm.fit(X_train, y_train)

# Predict classes of test data using trained SVM model
y_pred = svm.predict(X_test)

# Evaluate accuracy of SVM model on test data
accuracy = accuracy_score(y_test, y_pred)

# Calculate AUC score of SVM model on test data
auc = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'AUC score: {auc}')


# In[ ]:


from sklearn.svm import SVC
filename = f"/Users/kritika/Documents/data2/13features.txt"
#print('-'*100)
df = pd.read_csv(filename,sep = ',', header=None)
y = df.iloc[:, 0] #labels are in the first column
X = df.iloc[:, 1:] #features are in the rest of the columns
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.25)

# Instantiate SVM classifier with linear kernel
svm = SVC(kernel='linear')

# Train SVM model on training data
svm.fit(X_train, y_train)

# Predict classes of test data using trained SVM model
y_pred = svm.predict(X_test)

# Evaluate accuracy of SVM model on test data
accuracy = accuracy_score(y_test, y_pred)

# Calculate AUC score of SVM model on test data
auc = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'AUC score: {auc}')


# # Now I'm using all the scenarios as a whole by testing on some scenarios and training on the remaining

# In[ ]:


# Load the text files and create a list of dataframes
dataframes = []
for i in range(1, 14):
    filename = f"/Users/kritika/Documents/data2/{i}features.txt"
    df = pd.read_csv(filename,sep = ',', header=None)
    dataframes.append(df)
# Combine the dataframes into a single dataframe
df = pd.concat(dataframes, ignore_index=True)


# In[ ]:


y = df.iloc[:, 0] #labels are in the first column
X = df.iloc[:, 1:] #features are in the rest of the columns


# In[ ]:


# Set the number of folds for k-fold cross-validation
num_folds = 2


# In[ ]:


# Initialize the random forest classifier
rf = RandomForestClassifier(n_estimators=10)

# Initialize the k-fold cross-validation object
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)


# In[ ]:


# Initialize lists to store the accuracy scores and AUC scores
acc_scores = []
auc_scores = []

# Loop through the k-folds
for train_index, test_index in kf.split(X):
    
    # Split the data into training and test sets
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Fit the model on the training data
    rf.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = rf.predict(X_test)
    
    # Calculate the accuracy score
    acc_score = accuracy_score(y_test, y_pred)
    
    # Calculate the AUC score
    auc_score = roc_auc_score(y_test, y_pred)
    
    # Append the accuracy score to the list
    acc_scores.append(acc_score)
    
    # Append the AUC score to the list
    auc_scores.append(auc_score)
    
# Calculate the mean accuracy score
mean_acc_score = np.mean(acc_scores)

# Calculate the mean AUC score
mean_auc_score = np.mean(auc_scores)

# Print the mean accuracy score and mean AUC score
print('Mean Accuracy Score:', mean_acc_score)
print('Mean AUC Score:', mean_auc_score)

