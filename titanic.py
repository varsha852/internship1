#!/usr/bin/env python
# coding: utf-8

# # Titanic Project

# The Titanic Problem is based on the sinking of the ‘Unsinkable’ ship Titanic in the early 1912. It gives you information about multiple people like their ages, sexes, sibling counts, embarkment points and whether or not they survived the disaster. Based on these features, you have to predict if an arbitrary passenger on Titanic would survive the sinking.

# In[1]:


#pip install seaborn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


#Load the data
titanic = sns.load_dataset('titanic')
#Print the first 10 rows of data
titanic.head(10)


# Now, I will analyze the data by getting counts of data, survival rates, and creating charts to visualize the data.
# 
# Get a count of the number of rows and columns in the data set. Note that each row is a passenger onboard the ship and the columns are data points for each passenger.
# 
# There are 892 rows/passengers and 15 columns/data points in the data set.

# In[3]:


#Count the number of rows and columns in the data set 
titanic.shape


# In[4]:


#Get a count of the number of survivors 
titanic['survived'].value_counts()


# In[5]:


#Visualize the count of number of survivors
sns.countplot(titanic['survived'],label="Count")


# In[6]:


# Visualize the count of survivors for columns 'who', 'sex', 'pclass', 'sibsp', 'parch', and 'embarked'
cols = ['who', 'sex', 'pclass', 'sibsp', 'parch', 'embarked']

n_rows = 2
n_cols = 3

# The subplot grid and the figure size of each graph
# This returns a Figure (fig) and an Axes Object (axs)
fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*3.2,n_rows*3.2))

for r in range(0,n_rows):
    for c in range(0,n_cols):  
        
        i = r*n_cols+ c #index to go through the number of columns       
        ax = axs[r][c] #Show where to position each subplot
        sns.countplot(titanic[cols[i]], hue=titanic["survived"], ax=ax)
        ax.set_title(cols[i])
        ax.legend(title="survived", loc='upper right') 
        
plt.tight_layout()   #tight_layout


# In[7]:


#Look at survival rate by sex
titanic.groupby('sex')[['survived']].mean()


# In[8]:


#Look at survival rate by sex and class
titanic.pivot_table('survived', index='sex', columns='class')


# In[9]:


#Plot the survival rate of each class.
sns.barplot(x='class', y='survived', data=titanic)


# In[10]:


#Count the empty (NaN, NAN, na) values in each column 
titanic.isna().sum()


# In[14]:


#Split the data into independent 'X' and dependent 'Y' variables
X = titanic.iloc[:, 1:8].values 
Y = titanic.iloc[:, 0].values 


# In[15]:


# Split the dataset into 80% Training set and 20% Testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# In[16]:


#Look at all of the values in each column & get a count 
for val in titanic:
   print(titanic[val].value_counts())
   print()


# In[17]:


# Drop the columns
titanic = titanic.drop(['deck', 'embark_town', 'alive', 'class', 'alone', 'adult_male', 'who'], axis=1)

#Remove the rows with missing values
titanic = titanic.dropna(subset =['embarked', 'age'])


# In[18]:


#Count the NEW number of rows and columns in the data set
titanic.shape


# In[19]:


titanic.dtypes


# In[20]:


#Print the unique values in the columns
print(titanic['sex'].unique())
print(titanic['embarked'].unique())


# In[21]:


#Encoding categorical data values (Transforming object data types to integers)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

#Encode sex column
titanic.iloc[:,2]= labelencoder.fit_transform(titanic.iloc[:,2].values)
#print(labelencoder.fit_transform(titanic.iloc[:,2].values))

#Encode embarked
titanic.iloc[:,7]= labelencoder.fit_transform(titanic.iloc[:,7].values)
#print(labelencoder.fit_transform(titanic.iloc[:,7].values))

#Print the NEW unique values in the columns
print(titanic['sex'].unique())
print(titanic['embarked'].unique())


# In[29]:


#Split the data into independent 'X' and dependent 'Y' variables
X = titanic.iloc[:, 1:8].values 
Y = titanic.iloc[:, 0].values 


# In[30]:


# Split the dataset into 80% Training set and 20% Testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# In[31]:


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[46]:


#Create a function within many Machine Learning Models
def models(X_train,Y_train):
  
  #Using Logistic Regression Algorithm to the Training Set
  from sklearn.linear_model import LogisticRegression
  log = LogisticRegression(random_state = 0)
  log.fit(X_train, Y_train)
  
  #Using KNeighborsClassifier Method of neighbors class to use Nearest Neighbor algorithm
  from sklearn.neighbors import KNeighborsClassifier
  knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
  knn.fit(X_train, Y_train)

  #Using SVC method of svm class to use Support Vector Machine Algorithm
  from sklearn.svm import SVC
  svc_lin = SVC(kernel = 'linear', random_state = 0)
  svc_lin.fit(X_train, Y_train)

  #Using SVC method of svm class to use Kernel SVM Algorithm
  from sklearn.svm import SVC
  svc_rbf = SVC(kernel = 'rbf', random_state = 0)
  svc_rbf.fit(X_train, Y_train)

  #Using GaussianNB method of naïve_bayes class to use Naïve Bayes Algorithm
  from sklearn.naive_bayes import GaussianNB
  gauss = GaussianNB()
  gauss.fit(X_train, Y_train)

  #Using DecisionTreeClassifier of tree class to use Decision Tree Algorithm
  from sklearn.tree import DecisionTreeClassifier
  tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
  tree.fit(X_train, Y_train)

  #Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm
  from sklearn.ensemble import RandomForestClassifier
  forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
  forest.fit(X_train, Y_train)
  
  #print model accuracy on the training data.
  print('[0]Logistic Regression Training Accuracy:', log.score(X_train, Y_train))
  print('[1]K Nearest Neighbor Training Accuracy:', knn.score(X_train, Y_train))
  print('[2]Support Vector Machine (Linear Classifier) Training Accuracy:', svc_lin.score(X_train, Y_train))
  print('[3]Support Vector Machine (RBF Classifier) Training Accuracy:', svc_rbf.score(X_train, Y_train))
  print('[4]Gaussian Naive Bayes Training Accuracy:', gauss.score(X_train, Y_train))
  print('[5]Decision Tree Classifier Training Accuracy:', tree.score(X_train, Y_train))
  print('[6]Random Forest Classifier Training Accuracy:', forest.score(X_train, Y_train))
  
  return log, knn, svc_lin, svc_rbf, gauss, tree, forest


# In[47]:


#Get and train all of the models
model = models(X_train,Y_train)


# In[53]:


from sklearn.metrics import confusion_matrix 
for i in range(len(model)):
   cm=confusion_matrix(Y_test, model[i].predict(X_test)) 
   #extracting TN, FP, FN, TP
   TN, FP, FN, TP = confusion_matrix(Y_test, model[i].predict(X_test)).ravel()
   print('Model[{}] Testing Accuracy = "{} !"'.format(i,  (TP + TN) / (TP + TN + FN + FP)))
   print()# Print a new line


# In[54]:


#Print Prediction of Random Forest Classifier model
pred = model[6].predict(X_test)
print(pred)

#Print a space
print()

#Print the actual values
print(Y_test)


# Now that we have analyzed the data, created our models, and chosen a model to predict who would’ve survived the Titanic, let’s test and see if I would have survived.
# 
# I will create a variable called my_survival.
# 
# In it, I will have a pclass = 3, meaning I would probably be in the third class because of the cheaper price.
# I am a male, so sex = 1.
# I am older than 18, so I will put age = 21.
# Most likely, I would not be on the ship with siblings or spouses, so sibsp = 0.
# Nor with any children or parents, so parch = 0.
# I would try to pay the minimum fare, so fare = 0.
# I would’ve embarked from Queenstown, so embarked = 1.
# Putting those values in an array gives me [3,1,21,0, 0, 0, 1]. But, to put this into the prediction method of the model, it must be a list of lists or 2D array, for example [[3,1,21,0, 0, 0, 1]].

# In[58]:


my_survival = [[3,1,21,0, 0, 0, 1]]
#Print Prediction of Random Forest Classifier model
pred = model[6].predict(my_survival)
print(pred)

if pred == 0:
  print('you are dead')
else:
  print('Nice! You survived')


# Looks like I would not have survived the Titanic if I was on board.
# 
# 

# In[ ]:




