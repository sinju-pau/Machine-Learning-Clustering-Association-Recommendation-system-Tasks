
# coding: utf-8

# # CROSS VALIDATION ON GESTURE PHASE DATA IN PYTHON

# The dataset is composed by features extracted from 1 video out of 7 videos with people gesticulating, aiming at studying Gesture Phase Segmentation. More at https://archive.ics.uci.edu/ml/datasets/Gesture+Phase+Segmentation .
# 
# The dataset contains 18 numeric attributes (double), a timestamp and a class attribute (nominal). Features include the position of hands, wrists, head and spine of the user in each frame x, y, and z along with velocity and acceleration of hands and wrists
# 
# The task here is to classify each observation to the appropriate gesture phase and then determine the hyperparameters by cross-validation techniques.

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
plt.style.use('ggplot')
get_ipython().magic('matplotlib inline')


# Libraries well-imported, now lets load the dataset and give it a random shuffle.

# In[2]:

gesture = pd.read_csv('gesture.csv')


# In[3]:

gesture = gesture.sample(frac=1).reset_index(drop=True)
gesture.head(20)


# In[4]:

gesture.info()


# Observe that the dataset has 1747 rows with 20 columns. We include the variables from 0 to 17 in the feature space in X and 19th variable as the class/output

# In[5]:

X = gesture.iloc[:, 0:18].values
y = gesture.iloc[:,19].values


# ## Data Prepricessing

# Now perform feature scaling

# In[6]:

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)


# Also we need to encode categorical output variable, y

# In[7]:

# Encoding Categorical output
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)


# In[8]:

#Do the'train_test_split'
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# ## Building a classifier
# 
# Lets use the well-known SVM classifier on the data.

# In[9]:

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Now print the error metrics

# In[10]:

from sklearn import metrics
print('The accuracy of the svm',metrics.accuracy_score(y_pred,y_test))


# Now apply k-fold cross validation with k = 10 and determine the mean accuracy

# In[11]:

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()


# The mean accuracy is very close to our actual accuracy.

# Now apply the Grid Search to find the best accuracy and  hyperparameters such as 'C', 'kernel' and 'gamma'.

# In[12]:

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)


# To get the best accuracy score, mean of ten accuracies are evaluated. best_score  and best_params are the attributes to be used

# In[13]:

best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


# In[14]:

best_accuracy


# In[15]:

best_parameters


# ## Conclusion
# 1. The SVM Classifier has an accuracy of 87.2 % on the test set.The accuracy determined using CV is 87.3 %.
# 2. The best accuracy determined using GridSearch is 94 % and best parameters are :
#     {'C': 100, 'gamma': 0.4, 'kernel': 'rbf'}
