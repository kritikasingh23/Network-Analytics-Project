#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
import os, glob
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.decomposition import PCA


# In[31]:


filenames = [i for i in glob.glob(os.path.expanduser("~/Documents/data2/*.txt"))]


# In[32]:


len(filenames)


# In[33]:


filenames


# In[35]:


# Load the text files and create a list of dataframes
dataframes = []
for i in range(1, 14):
    filename = f"/Users/kritika/Documents/data3/11features.txt"
    df = pd.read_csv(filename,sep = ',', header=None)
    dataframes.append(df)
# Combine the dataframes into a single dataframe
df = pd.concat(dataframes, ignore_index=True)


# In[28]:


from pandas_profiling import ProfileReport
prof = ProfileReport(df)
prof


# In[36]:


df.shape


# In[37]:


df.head()


# In[38]:


y = df.iloc[:, 0] #labels are in the first column
X = df.iloc[:, 1:] #features are in the rest of the columns


# In[39]:


X.shape


# In[40]:


y.shape


# In[41]:


y.value_counts()


# In[45]:


from sklearn.model_selection import StratifiedKFold
# Reduce the dimensionality of the data using PCA
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)

# Initialize the stratified k-fold cross-validator
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Train and evaluate the model on each split of the data
for train_index, test_index in skf.split(X_pca, y):
    X_train, X_test = X_pca[train_index], X_pca[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Initialize the random forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the random forest classifier on the training data
    rf.fit(X_train, y_train)
    
    # Predict the labels for the test data
    y_pred = rf.predict(X_test)
    # Calculate the accuracy of the predictions
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")


# In[16]:


from sklearn.model_selection import StratifiedKFold
# Initialize the stratified k-fold cross-validator
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# In[17]:


# Split the data into training and testing sets using stratified k-fold
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # train and evaluate your model on this split


# In[24]:


# Initialize the random forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
# Train the random forest classifier on the training data
rf.fit(X_train, y_train)
    
# Predict the labels for the test data
y_pred = rf.predict(X_test)
    
# Calculate the accuracy of the predictions
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


# In[20]:


# Initialize the random forest classifier
rf = RandomForestClassifier(n_estimators=10)


# In[13]:


# Set the number of folds for k-fold cross-validation
num_folds = 2


# In[14]:


# Initialize the random forest classifier
rf = RandomForestClassifier(n_estimators=10)

# Initialize the k-fold cross-validation object
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)


# In[15]:


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


# In[15]:


# Initialize the Loocv object
loo = LeaveOneOut()


# In[16]:


# Loop through the loco
for train_index, test_index in loo.split(X):
    
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


# In[17]:


acc_scores


# In[18]:


mean_acc_score


# In[19]:


auc_scores


# In[20]:


mean_auc_score


# In[ ]:




