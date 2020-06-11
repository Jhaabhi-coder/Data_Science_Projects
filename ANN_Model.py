# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 00:04:44 2020

@author: Abhijeet
"""

# Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importing the dataset

dataset = pd.read_csv('Churn_Modelling.csv')
pd.set_option('display.max_columns', 500)
dataset.head()
dataset.columns
dataset.isna().any()
user_identifier = dataset['CustomerId']

# Part-1 --- Visualizing the dataset

## Histogram

dataset2 = dataset.drop(columns=['CustomerId', 'Surname', 'Exited', 'RowNumber'])

fig = plt.figure(figsize=(15, 12))
plt.suptitle("Histogram of numrical values", fontsize = 15)
for i in range(1, dataset2.shape[1] + 1):
    plt.subplot(4, 3, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(dataset2.columns.values[i - 1])
    
    vals = np.size(dataset2.iloc[:, i - 1].unique())
    plt.hist(dataset2.iloc[:, i - 1], bins = vals, color = '#51238c')
    
plt.tight_layout(rect = [0, 0.03, 1, 0.95])

    
## Pie Chart

fig = plt.figure(figsize = (15, 12))
plt.suptitle('Pie_Chart', fontsize = 15)

for i in range(1, dataset2.shape[1] + 1):
    plt.subplot(4, 3, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(dataset2.columns.values[i - 1])
    
    values = dataset2.iloc[:, i - 1].value_counts(normalize = True).values
    index = dataset2.iloc[:, i - 1].value_counts(normalize = True).index
    
    plt.pie(values, autopct= '%1.1f%%', labels = index)
    plt.axis('equal')

plt.tight_layout(rect = [0, 0.03, 1, 0.95])

## Correlation Metrix

correl = dataset.drop(columns = ['CustomerId', 'Surname', 'RowNumber'])
plt.subplots(figsize=(30,25))                        
sns.heatmap(correl.corr(), annot = True)

# Part-2 ---Data Preprocessing

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


# Encoding the categorical variables

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
Encode_X_1 = LabelEncoder()
X[:, 1] = Encode_X_1.fit_transform(X[:, 1])
Encode_X_2 = LabelEncoder()
X[:, 2] = Encode_X_2.fit_transform(X[:, 2])
ct = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype = np.float)
X = X[:, 1:]

# Spliting the dataset into training and test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

   
# Feature Scaling Independent Variable

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Part-3--- Bulding Artificial Neural Network (ANN)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initializing the ANN

classifier = Sequential()

# Adding Input layer and frist hidden layer

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dropout(p = 0.1))

    
# Adding Second hidden layer

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.1))

# Adding the output layer

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
# Fitting the ANN to Traning 
import time

t0 = time.time()
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
t1 = time.time()

print('Took %0.2f seconds' % (t1-t0))

# Predicting result on test set

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Evaluating the result

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)


# Random Prediction (Out of Sample)

'''Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000'''

new_pred = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_pred = (new_pred > 0.50)


# K-Fold Validation

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
t0 = time.time()
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
t1 = time.time()
print('Took almost %0.2f seconds' % (t1-t0))

mean = accuracies.mean()
variance = accuracies.std()


# Grid Search
t0 = time.time()
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)

parameters = {'batch_size': [25, 32],
              'nb_epoch': [100, 500],
              'optimizer': ['adam','rmsprop']}

grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10)
grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
t1 = time.time()

print('Took %0.2f seconds' % (t1-t0))








