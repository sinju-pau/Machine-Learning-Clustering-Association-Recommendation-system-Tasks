
# coding: utf-8

# # DIMENSIONALITY REDUCTION : PCA and LDA ON WINE DATA
# 
# Dimensionality reduction refers to feature extraction technique used widely in Machine Learning Approaches to reduce the number of features for analysis. We concentrate on two such tools :
# 
#    1. Principal Component Analysis (PCA)
#    2. Linear Discriminant Analysis (LDA)
# 
# PCA is a useful unsuperwised statistical technique that has found application in fields such as face recognition and image compression, and is a common technique for finding patterns in data of high dimension. PCA extrats p <= m  of m independent variables that explain most of the variance of the dataset, regardless of the independent variable. 
# 
# LDA is a superwised technique that extracts p <= m  of m independent variables that separate the most the classes of the dependent variable. 
# 
# Here we perform Dimensionality Reduction techniques and analysis on the wine data from http://archive.ics.uci.edu/ml/datasets/Wine?ref=datanews.io.
# 
# This is a classification problem involving many input features and we are trying to reduce the number of such features. The task is to use chemical analysis to determine the origin of wines from three classes. The anlaysis is done in Python 3

# Begin by importing the libraries

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
plt.style.use('ggplot')
get_ipython().magic('matplotlib inline')


# Now read the dataset and view it

# In[2]:

wine = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header =None)


# In[3]:

wine.head(10)


# In[4]:

wine.info()


# There are no NA's, now assign column names, also give a random shuffle to the dataste to reflect all class instances.

# In[5]:

wine.columns = ["Class","Alcohol","Malic_acid","Ash","Alcalinity_of_ash","Magnesium","Total_phenols",
                "Flavanoids","Nonflavanoid_phenols","Proanthocyanins","Color_intensity",
                "Hue","OD280/OD315","Proline"]
wine = wine.sample(frac=1).reset_index(drop=True)
wine.head(10)


# In[6]:

X = wine.iloc[:, 1:14].values
y = wine.iloc[:,0].values


# ## 1. Principal Component Analysis 
# 
# Lets move to PCA. Start by splitting data into training and test sets and perform feature scaling.

# In[7]:

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[8]:

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[9]:

# Applying PCA to reduce the number of features
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


# Above,we have implemented the PCA with n_components = 2 i.e. we want the PCA object to output two principal components that can explain most of the variance of data in the model.

# In[10]:

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# The predictions are good enough for the data. Accuracy on the test set  is 100 %.
# 
# Lets analyze the explained variance score

# In[11]:

explained_variance = pca.explained_variance_ratio_
explained_variance


# Observe that the two principal components retured are able to explain the variance of the data by 0.36 + 0.18 = 0.54 or 54 %

# ## Data Visualization
# 
# Now that the number of input features are reduced to 2, we can visualize how the Logistic Regression has done the classification. We have used the ListedColormap class from matplotlib library to create the contours as well as the scatter plots.

# In[12]:

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('pink', 'lightgreen', 'lightblue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()


# In[13]:

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('pink', 'lightgreen', 'lightblue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()


# ## 2. Linear Discriminant Analysis
# 
# As stated earlier, LDA worries about the separability of the classes rather than the variance and hence, it is a superwised approach that makes use of the target variable, class.

# In[18]:

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[19]:

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[20]:

# Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, y_train) #its a supervised algorithm, so include DV for fitting too.
X_test = lda.transform(X_test)


# In[22]:

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# In[29]:

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
explained_variance = lda.explained_variance_ratio_
explained_variance


# We observe that predictions are good enough and the LDA can separate the classes by 0.70 +0.30 = 100 %. Lets now visualize the results for LDA

# ## Data Visualization

# In[25]:

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('pink', 'lightgreen', 'lightblue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Linear Discriminant 1')
plt.ylabel('Linear Discriminant 2')
plt.legend()
plt.show()


# In[28]:

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('pink', 'lightgreen', 'lightblue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Linear Discriminant 1')
plt.ylabel('Linear Discriminant 2')
plt.legend()
plt.show()


# ## Concluding Remarks
# 
# 1. For the wine dataset, the two principal components generated are able to explain the variance of the data by 54 %
# 2. LDA can separate the classes by 100 %, this is because the classes are almost well-separable
# 3. For data with classes not well-separable, Kernel-PCA approach is suggested.
# 
