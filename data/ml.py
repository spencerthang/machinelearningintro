import scipy
import numpy as np
import pandas as pd
import plotly.plotly as py

#import visplots

from plotly.graph_objs import *
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from sklearn import preprocessing, metrics
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from scipy.stats.distributions import randint

data = pd.read_csv('../data/wine_quality.csv')
data_numpy = np.array(data)

x = data_numpy[:,:-1].astype(float)
y = data_numpy[:, -1]
x = preprocessing.scale(x)

# Convert to numbered classifiers
le = preprocessing.LabelEncoder()
y  = le.fit_transform(y)

XTrain, XTest, yTrain, yTest = train_test_split(x, y, random_state=1)

# Create knn3
knn3 = KNeighborsClassifier(n_neighbors=3, weights='distance')
knn3.fit(XTrain, yTrain)
yPredict = knn3.predict(XTest)
print metrics.confusion_matrix(yTest, yPredict)


# knn3 cross validation score
knn3scores = cross_val_score(knn3, x, y, cv = 5)
print(knn3scores)
print("Mean of scores KNN3", knn3scores.mean())

# Grid search with 10-fold cross-validation using a dictionary of parameters
# Create the dictionary of given parameters
n_neighbors = np.arange(1, 51, 2)
weights     = ['uniform','distance']
parameters  = [{'n_neighbors': n_neighbors, 'weights': weights}]

# Optimise and build the model with GridSearchCV
gridCV = GridSearchCV(KNeighborsClassifier(), parameters, cv=10)
gridCV.fit(XTrain, yTrain)

# Report the optimal parameters
bestNeighbors = gridCV.best_params_['n_neighbors']
bestWeight    = gridCV.best_params_['weights']

print("Best parameters: n_neighbors=", bestNeighbors, "and weight=", bestWeight)

# Build the best kNN classifier
knn = KNeighborsClassifier(n_neighbors=bestNeighbors, weights=bestWeight)
knn.fit(XTrain, yTrain)
yPredKnn = knn.predict(XTest)

# Report the test accuracy and performance
print(metrics.classification_report(yTest, yPredKnn))
print("Overall Accuracy:", round(metrics.accuracy_score(yTest, yPredKnn), 2))

# Random search randomly tests a larger set of neighbours


# Decision Tree gridSearch
# Conduct a grid search with 10-fold cross-validation using the dictionary of parameters
n_estimators = np.arange(1, 30, 5)
max_depth    = np.arange(1, 100, 5)
parameters   = [{'n_estimators': n_estimators, 'max_depth': max_depth, 'random_state': [0]}]

gridCV = GridSearchCV(RandomForestClassifier(), param_grid=parameters, cv=10)
gridCV.fit(XTrain, yTrain)

# Print the optimal parameters
best_n_estim   = gridCV.best_params_['n_estimators']
best_max_depth = gridCV.best_params_['max_depth']

print ("Best parameters: n_estimators=", best_n_estim,", max_depth=", best_max_depth)

clfRDF = RandomForestClassifier(n_estimators=best_n_estim, max_depth=best_max_depth)
clfRDF.fit(XTrain, yTrain)
predRF = clfRDF.predict(XTest)

print(metrics.classification_report(yTest, predRF))
print("Overall Accuracy:", round(metrics.accuracy_score(yTest, predRF),2))