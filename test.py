import pandas as pd
import numpy as np
import matplotlib as plt
#%matplotlib inline
import matplotlib.pyplot as plt
import sklearn

df = pd.read_csv("loands.csv")
print(df)
print(df.describe())
df['LoanAmount'].hist(bins=50)
df['ApplicantIncome'].hist(bins=50)


temp1 = df
fig= plt.figure(figsize=(8,4))
ax1=fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Applicants')
ax1.set_xlabel('Applicants by Credit_History')
#temp1.plot(kind='bar')


df['LoanAmount'].fillna(df['LoanAmount'].mean(),inplace= True) #filling missing values
print(df.apply(lambda x:sum(x.isnull()),axis=0)) #number of missing values in each column
print(df.mean()) #basic math operations df.median,cumulative summation etc
print(df.dtypes)

df['Credit_History'].fillna(df['Credit_History'].mean(),inplace= True) #filling missing values
print(df.apply(lambda x:sum(x.isnull()),axis=0)) #number of missing values in each column
print(df.mode()) #basic math operations df.median,cumulative summation etc
#plt.show()

one = pd.DataFrame(np.random.randn(5,4))
print(one)

two = pd.DataFrame(np.random.randn(5,4))
print(two)
print(pd.concat([one,two]))

left = pd.DataFrame({'key':['foo','bar'],'lval':[1,2]})
print(left)

right = pd.DataFrame({'key':['foo','bar','bar'],'rval':[3,4,5]})
print(right)

print(pd.merge(left,right,on='key'))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn import metrics


X= df.iloc[:,[8,10]].values
y= df.iloc[:,12].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
print(classifier.fit(X_train,y_train))
y_pred = classifier.predict(X_test)
print(y_pred)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=0)
print(tree.fit(X_train,y_train))
y1_pred = tree.predict(X_test)
print(y1_pred)
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test,y1_pred)
print(cm1)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y1_pred))


from sklearn.naive_bayes import MultinomialNB
naive1 = MultinomialNB()
print(naive1.fit(X_train,y_train))
y2_pred = naive1.predict(X_test)
print(y2_pred)
cm2 = confusion_matrix(y_test,y2_pred)
print(cm2)
print(accuracy_score(y_test,y2_pred))



from sklearn.ensemble import RandomForestClassifier
tree1 = RandomForestClassifier(random_state=0)
print(tree1.fit(X_train,y_train))
y3_pred = tree1.predict(X_test)
print(y3_pred)
cm3 = confusion_matrix(y_test,y3_pred)
print(cm3)


print(accuracy_score(y_test,y_pred))
print(accuracy_score(y_test,y1_pred))
print(accuracy_score(y_test,y2_pred))
print(accuracy_score(y_test,y3_pred))



# # series
# x = pd.Series([6,3,4,6])
#
# # data frame
# df= pd.DataFrame(np.random.randn(4,3))
