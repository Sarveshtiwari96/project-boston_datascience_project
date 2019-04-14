import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df_1 = pd.read_csv('boston.csv')
print(df_1.head())

df = df_1.drop("ID",axis=1)


col_names = df_1.columns.values
print(col_names)
X_1 = df_1[col_names[0:14]]
X= X_1.drop('ID',axis=1)
print(X)
Y = df_1[col_names[-1]]
print(Y)

corr = df.corr()
print(corr)

# sns.heatmap(corr,annot=True)

# plt.show()


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.7,random_state=1)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
r = lr_model.fit(x_train,y_train)
print(r)
print(r.coef_)
print(r.intercept_)
model_predict = lr_model.predict(x_test)
print(model_predict)


from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test,model_predict))

from sklearn.metrics import r2_score
print(r2_score(y_test,model_predict))


x1= df.iloc[:,[5,12]]
y1= df.iloc[:,-1]


x_train,x_test,y_train,y_test = train_test_split(x1,y1,train_size=0.7,random_state=1)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
r = lr_model.fit(x_train,y_train)
print(r)
print(r.coef_)
print(r.intercept_)
model_predict = lr_model.predict(x_test)
print(model_predict)

from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test,model_predict))

from sklearn.metrics import r2_score
print(r2_score(y_test,model_predict))

