# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_profiling 


dataset = pd.read_csv(r'Train_dataset_2.csv') #(use "r" before the path string to address special character, such as '\'). Don't forget to put the file name at the end of the path + '.xlsx'
dataset_2 = pd.read_csv('Test_dataset.csv')

X2 = dataset.iloc[:, 1:].values
Xtest = dataset_2.iloc[:, 12:13].values

'''
#Profiling data
profile = dataset.profile_report(title='Diuresis Profiling Report')
profile.to_file(output_file="Duresis profiling_2.html")
'''

#Data Preprocessing
X = X2
for i in range(7):
    for j in range(10714):
        #X[j][i] = int(X[j][i])
        if isinstance(X[j][i],int):
            f=0
        else:
            if (len(X[j][i])) == 7:
                c=1000*int(X[j][i][1])+100*int(X[j][i][3])+10*int(X[j][i][4])+1*int(X[j][i][5])
                X[j][i]=c
            else:
                c=100*int(X[j][i][1])+10*int(X[j][i][2])+1*int(X[j][i][3])
                X[j][i]=c

X2 = X
X=[]

for i in range(6):
    for j in  range(10714):
        X.append([X2[j][i],X2[j][i+1]])
X = np.asarray(X)
y= X[:,1]
X = X[:,0:1]


#Splitting Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#Fitting RandomForestRegressor 
from sklearn.ensemble import RandomForestRegressor
ts_regressor = RandomForestRegressor(n_estimators = 1000,random_state = 45)
#regressor.fit(X_train,y_train)
ts_regressor.fit(X_train,y_train)
# Predicting a new result
y_pred = ts_regressor.predict(X_test)
from sklearn.metrics import mean_squared_error
mse_diuresis = mean_squared_error(y_test,y_pred)

plt.plot(y_test,y_pred, color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression Model)')
plt.xlabel('Real')
plt.ylabel('Predicted')
plt.savefig('Diuresis_predictions using Random Forest')
plt.show()

#Predicting the the diuresis values on 27th March
# using the diuresis values on 20th March as given in Test_datasset
y_pred=[]

date = 21
for i in range(7):
    print(date)
    Xtest = ts_regressor.predict(Xtest)
    Xtest = Xtest.reshape(14498,1)
    date = date +1
pd.DataFrame(Xtest, columns=['Diuresis']).to_csv('Pred_2_ran_for_final.csv')



