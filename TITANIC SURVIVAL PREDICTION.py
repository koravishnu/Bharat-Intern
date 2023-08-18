#!/usr/bin/env python
# coding: utf-8

# Titanic Survival Prediction with Python

# In[ ]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
get_ipython().run_line_magic('', 'matplotlib in line')


# In[3]:


titanic = pd.read_csv('train.csv')
titanic.head()


# In[4]:


titanic.shape


# In[6]:


sns.countplot(x='Survived',data=titanic)


# In[7]:


#those who did not survived(more than 500)are greater than those who survived(nearly 300)


# In[8]:


sns.countplot(x='Survived',hue='Sex',data=titanic,palette='winter')


# In[9]:


## Analysis:0 represents not survived and 1 is for survives
#women are thrice more likely to survive than males.


# In[11]:


sns.countplot(x='Survived',hue='Pclass',data=titanic,palette='PuBu')


# In[12]:


##Analysis:the passengers who did not survived belong to the 3rd class
##1st class passengers are more likely to survive


# In[13]:


titanic['Age'].plot.hist()


# In[14]:


#we notice that highest age group travelling are among the young age between 20-40.
#very few passengers in age group 70-80


# In[15]:


titanic['Fare'].plot.hist(bins=20,figsize=(10,5))


# In[16]:


#we observe that most of the tickets bought are under fare 100
#and very few are on the higher side of fare i.e. 220-500 range


# In[17]:


sns.countplot(x='SibSp',data=titanic,palette='rocket')


# In[18]:


#we notice that most of the passengers do not have their sibilings abroad.


# In[20]:


titanic['Parch'].plot.hist()


# In[21]:


sns.countplot(x='Parch',data=titanic,palette='summer')


# In[22]:


#Data Wrangling means cleaning the data,removing the null vales,
#dropping unwanted columns, adding new ones if needed.


# In[23]:


titanic.isnull().sum()


# In[24]:


#age and cabin has most null vales,and embarked too has null vales
#we can plot it on heat map


# In[25]:


sns.heatmap(titanic.isnull(),cmap='spring')


# In[26]:


#here yellow color is showing the null values,highest in cabin followed by age


# In[27]:


sns.boxplot(x='Pclass',y='Age',data=titanic)


# In[28]:


#we can observe that older age group are travelling more in class 1 and 2
#compared to class 3


# In[29]:


#the hue parameter determines which column in the data frame should be used for colour encoding


# In[30]:


#we will drop a few columns now


# In[31]:


titanic.head()


# In[34]:


titanic.drop('Cabin',axis=1,inplace=True)


# In[35]:


titanic.head(3)#dropped the cabin column


# In[36]:


titanic.dropna(inplace=True)


# In[37]:


sns.heatmap(titanic.isnull(),cbar=False)


# In[38]:


#this shows that we dont have any null vales we can also check it:


# In[39]:


titanic.isnull().sum()


# In[40]:


titanic.head(2)


# In[41]:


#we will convert the few columns(strings)into categorical data to apply logistic regression


# In[42]:


pd.get_dummies(titanic['Sex']).head()


# In[44]:


sex=pd.get_dummies(titanic['Sex'],drop_first=True)
sex.head(3)


# In[45]:


##we have dropped the first column because only one column is suffient to determine
#the gender of the passenger either will be male(1) or not(0),that means a female


# In[46]:


embark=pd.get_dummies(titanic['Embarked'])


# In[47]:


embark.head(3)


# In[48]:


#c stands for cherbourg, Q for questions , S for southhampton
#we can drop any one of the column as we inter from the two columns itself


# In[49]:


embark=pd.get_dummies(titanic['Embarked'],drop_first=True)


# In[50]:


embark.head(3)


# In[51]:


#if both values are of 0 then passenger is travelling in 1 st class


# In[53]:


Pcl=pd.get_dummies(titanic['Pclass'],drop_first=True)
Pcl.head(3)


# In[54]:


#our data  is now converted into categorical data


# In[55]:


titanic=pd.concat([titanic,sex,embark,Pcl],axis=1)


# In[56]:


titanic.head(3)


# In[57]:


#deleting the unwanted columns


# In[60]:


titanic.drop(['Name','PassengerId','Pclass',"Ticket",'Sex',"Embarked"],axis=1,inplace=True)


# In[61]:


titanic.head(3)


# Train Data

# In[103]:


X=titanic.drop(['Survived'],axis=1)
y=titanic['Survived']


# In[63]:


from sklearn.model_selection import train_test_split


# In[76]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=4)


# Logistic Regression

# In[116]:


model=LogisticRegression()


# In[117]:


model.fit(X_train, y_train)            


# In[80]:


prediction=model.predict(X_test)


# In[88]:


prediction=lm.predict(X_test)


# In[90]:


from sklearn.metrics import classification_report


# In[91]:


from sklearn.metrics import classification_report


# In[98]:


from sklearn.metrics import confusion_matrix


# In[99]:


confusion_matrix(y_test,prediction)


# In[101]:


from sklearn.metrics import accuracy_score


# In[102]:


accuracy_score(y_test,prediction)


# In[ ]:




