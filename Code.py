import numpy as np
import pandas as pd
import sklearn as sk
import time
import torch
import matplotlib as plt

from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

#You need to download the dataset and then pull it from your local desktop
# You need to edit the line below
data = pd.read_csv('C:Put_Your_File_Path_Here/diabetic_data.csv')

#Only run one of these at a time

#y = data['gender']
#y.value_counts().plot.bar(x='Gender', y='Count')

#y = data['age']
#y.value_counts().plot.bar(x='Gender', y='Count')

#y = data['race']
#y.value_counts().plot.bar(x='Gender', y='Count')

#y = data['readmitted']
#y.value_counts().plot.bar(x='Gender', y='Count')

#Since we are primarily interested in factors that lead to early readmission, we defined the readmission attribute (outcome) as having two values: “readmitted,” if the patient was readmitted within 30 days of discharge or 
#“otherwise,” which covers both readmission after 30 days and no readmission at all.
#The above line was taken from the paper using this data set which can be read at https://www.hindawi.com/journals/bmri/2014/781670/#materials-and-methods
labelDictionary = {'NO' : 1, '<30' : 0,'>30' : 1}
#Data cleansing:
#encounter_id: random number that we should not take into account
#patient_nbr: random number that we should not take into account
#payer_code: how the patient pays should not be taken into account

old_data = data

data = data.drop(columns=['encounter_id', 'patient_nbr', 'payer_code','weight', 'acetohexamide', 'max_glu_serum', 'repaglinide', 'nateglinide', 'chlorpropamide', 'tolbutamide', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'glyburide-metformin', 'glipizide-metformin', 'metformin-rosiglitazone', 'metformin-pioglitazone' ])
data = data.replace({'readmitted': labelDictionary})
data = data.to_numpy()

#Get labels and remove labels column from data

rows, columns = data.shape
labels = data[:,columns-1]
labels=labels.astype('int')
data = data[:,:columns-1]

#Create the OneHotEncoder
enc = OneHotEncoder()
enc.fit(data)


data = enc.transform(data).toarray()
data = np.array(data)

#Split the data into a 80/20 split of training and testing data
x_train,x_test,y_train,y_test = train_test_split(data,labels,test_size=.2)

#Create SGDClassifier
SGD = make_pipeline(StandardScaler(), SGDClassifier(max_iter=2000))

#Training
tm = time.time()
SGD.fit(x_train, y_train)
print('Training time: ', time.time() - tm)

#Scoring
tm = time.time()
print(SGD.score(x_test,y_test))
print('Scoring time: ', time.time() - tm)

#SGDClassifier with Kernel approximization
rbf_feature = RBFSampler(gamma=1, random_state=1)
X_features = rbf_feature.fit_transform(x_train)
X_TestFeatures = rbf_feature.fit_transform(x_test)

SGDKernel = SGDClassifier()

tm = time.time()
SGDKernel.fit(X_features, y_train)
print('Training time: ', time.time() - tm)

#Scoring
tm = time.time()
print('Score: ',SGDKernel.score(X_TestFeatures, y_test))
print('Similarity to other SGD: ',SGDKernel.score(X_TestFeatures, SGD.predict(x_test)))
print('Scoring time: ', time.time() - tm)

#create Decision Tree
clf = DecisionTreeClassifier()

#Training
tm = time.time()
clf.fit(x_train,y_train)
print('Training Time: ', time.time()-tm)

#Predict
tm = time.time()
y_pred = clf.predict(x_test)
print('Testing Time: ',time.time()-tm)

#Print accuracy
print('Accuracy: ', metrics.accuracy_score(y_test,y_pred))
print('Similarity to SGD: ', metrics.accuracy_score(y_pred,SGD.predict(x_test)))
print('Accuracy Similarity to SGD Kernel: ', metrics.accuracy_score(y_pred,SGDKernel.predict(X_TestFeatures)))
