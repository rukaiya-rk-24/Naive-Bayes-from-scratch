#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[18]:


def fit(X_train, Y_train):
    result = {}
    #Getting all possible values for output
    class_values = set(Y_train)
    for current_class in class_values:
        result[current_class] = {}
        result["total_data"] = len(Y_train)
        current_class_rows = (Y_train == current_class) #This will give back a true false array.
        X_train_current = X_train[current_class_rows]  #We will get onsly those rows where we have a true for xi val.
        Y_train_current = Y_train[current_class_rows]
        #Getting number of columns
        num_features = X_train.shape[1]
        result[current_class]["total_count"] = len(Y_train_current)
        for j in range(1, num_features + 1): #Keeping feature names from 1 to n
            result[current_class][j] = {}
            #Getting all possible values for feature j
            all_possible_values = set(X_train[:, j - 1]) #j-1 because our array x_train is still 0_indexed
            for current_value in all_possible_values:
                result[current_class][j][current_value] = (X_train_current[:, j - 1] == current_value).sum()
                
    return result


# In[42]:


def probability(dictionary, x, current_class):
    output = np.log(dictionary[current_class]["total_count"]) - np.log(dictionary["total_data"])
    num_features = len(dictionary[current_class].keys()) - 1  #-1 because we have one key corresponding to total count
    for j in range(1, num_features + 1):
        xj = x[j - 1]
        count_current_class_with_value_xj = dictionary[current_class][j][xj] + 1
        count_current_class = dictionary[current_class]["total_count"] + len(dictionary[current_class][j].keys())
        #This len will correspond to Laplace correction
        current_xj_probablity = np.log(count_current_class_with_value_xj) - np.log(count_current_class)
        output = output + current_xj_probablity
    return output


# In[40]:


def predictSinglePoint(dictionary, x):
    classes = dictionary.keys()
    best_p = -1000
    best_class = -1
    first_run = True
    for current_class in classes:
        if (current_class == "total_data"):
            continue
        p_current_class = probability(dictionary, x, current_class)
        if (first_run or p_current_class > best_p):
            best_p = p_current_class
            best_class = current_class
        first_run = False
    return best_class

        


# In[33]:


# def predict(dictionary, X_test):
#     y_pred = []
#     for x in X_test:
#         x_class = predictSinglePoint(dictionary, x)
#         y_pred.append(x_class)
#     return y_pred
def predict(dictionary,x_test):
    y_pred = []
    for x in x_test:
        x_class = predictSinglePoint(dictionary, x)
        y_pred.append(x_class)
    return y_pred


# In[9]:


def makeLabelled(column):
    second_limit = column.mean()
    first_limit = 0.5 * second_limit
    third_limit = 1.5*second_limit
    for i in range (0,len(column)):
        if (column[i] < first_limit):
            column[i] = 0
        elif (column[i] < second_limit):
            column[i] = 1
        elif(column[i] < third_limit):
            column[i] = 2
        else:
            column[i] = 3
    return column


# In[10]:


from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
Y = iris.target


# In[11]:


for i in range(0,X.shape[-1]):
    X[:,i] = makeLabelled(X[:,i])


# In[12]:


from sklearn import model_selection
X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size=0.25,random_state=0)


# In[44]:


dictionary = fit(X_train,Y_train)
print(dictionary)


# In[43]:


y_pred = predict(dictionary,X_test)


# In[37]:


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(Y_test,y_pred))
print(confusion_matrix(Y_test,y_pred))


# In[38]:


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
print(classification_report(Y_test,Y_pred))
print(confusion_matrix(Y_test,Y_pred))


# In[ ]:




