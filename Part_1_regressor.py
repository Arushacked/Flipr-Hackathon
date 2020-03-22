# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_profiling 


#Loading dataset
dataset = pd.read_csv(r'Train_dataset.csv') #(use "r" before the path string to address special character, such as '\'). Don't forget to put the file name at the end of the path + '.xlsx'
dataset_2 = pd.read_csv('Test_dataset_updated_27.csv')
dataset = dataset.where((pd.notnull(dataset)), None)

people_id = dataset_2.iloc[:, 0].values
Xtest = dataset_2.iloc[:, 1:].values
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 23:24].values

'''
#Profiling data
profile = dataset.profile_report(title='Pandas Profiling Report')
profile.to_file(output_file="Covid data profiling_2.html")

profile = dataset_2.profile_report(title='Pandas Profiling Report(Test_data)')
profile.to_file(output_file="Covid data profiling for test data_2.html")
'''

'''
#DELETING FOLLOWING
Designation(Highly Corelated to gender)
name -> No significance
Region(Highly related to d/M and Not unique values in train/test datasets)
PulmoPulmonary_score (Hihgly Related to cardio_pressure)

'''
#Data preprocessing

#Missing values
col =[2,11,12,13,14,15,16,19,21] #Missing continuous values
cat_cols=[3,4,7,10] #Missing Catagorical values
#col= [2,3,7,11,12,14,15,19,21]
from sklearn.impute import SimpleImputer
j=0
imp=[]
for i in col:
    x = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.append(x)
    j=j+1
j=0
for i in col:
    imp[j] = imp[j].fit(X[:,[i]])
    X[:,[i]] = imp[j].transform(X[:,[i]])

#Categorical Missing values
j=0
cat_col = [3,4,7,10]
imp2=[]
for i in cat_col:
    x = SimpleImputer(missing_values=None, strategy="most_frequent")
    imp2.append(x)
    j=j+1
j=0
for i in cat_col:
    #imp2[j] = imp2[j].fit(X[:,[i]])
    #X[:,[i]] = imp2[j].transform(X[:,[i]])
    X[:,[i]] = imp2[j].fit_transform(X[:,[i]])
'''
mul_cat_cols =[3,4,7,10]
df = pd.DataFrame(data=X[:,mul_cat_cols],  columns=["column1", "column2","column3", "column4"])

#Profiling
profile = df.profile_report(title='Pandas Profiling Report(categories)')
profile.to_file(output_file="Covid data profiling_3_updated.html")
'''
#Conclusion
#3 & 7 still contains missing values


#Label Encoding.....
cat_cols =[0,1,3,4,7,10]

#Encoding true_false attributes
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
j=0
Le=[]
true_false_cols = [0,1]
for i in true_false_cols:
    x = LabelEncoder()
    Le.append(x)
    j=j+1
j=0
for i in true_false_cols:
    X[:,i] = Le[j].fit_transform(X[:,[i]])
    Xtest[:,i] = Le[j].transform(Xtest[:,[i]])
    j=j+1
'''
#Encoding multivalue attributes
mul_cat_cols =[3,4,7,10]
j=0
Le2=[]
for i in mul_cat_cols:
    x = LabelEncoder()
    Le2.append(x)
    j=j+1
j=0
for i in mul_cat_cols:
    X[:,i] = Le2[j].fit_transform(X[:,[i]])
    Xtest[:,i] = Le2[j].transform(Xtest[:,[i]])
    j=j+1
'''

#categorical_features = [0],
mul_cat_cols =[3,4,7,10]
#Working Here
mul_cat_cols =[3,4,7,10]
j=0
ohc=[]
for i in mul_cat_cols:
    x = OneHotEncoder(drop='first')
    ohc.append(x)
    j=j+1

j=0
for i in mul_cat_cols:
    c = ohc[j].fit_transform(X[:,i:i+1]).toarray()
    d = ohc[j].transform(Xtest[:,i:i+1]).toarray()
    if j>0:
        test2 = np.append(test2, d, axis = 1)
        test = np.append(test, c, axis = 1)
    else:
        test = c
        test2 = d
    j=j+1
X = np.delete(X,obj=mul_cat_cols,axis=1)
X = np.append(X, test, axis = 1)
Xtest = np.delete(Xtest,obj=mul_cat_cols,axis=1)
Xtest = np.append(Xtest, test2, axis = 1)

mse_errors = []

#Splitting Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


#Fitting Regressor
model_name = 'Random Forest'
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 1000,random_state = 45)
regressor.fit(X_train,y_train)


# Predicting a new result
y_pred = regressor.predict(X_test)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,y_pred)
mse_errors.append([model_name,mse])
plt.scatter(y_test,y_pred, color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression Model)')
plt.xlabel('Real')
plt.ylabel('Predicted')
plt.show()

#Predicting on Test_datasset and saving
y_ran_for = regressor.predict(Xtest)

y_final = []
for i in range(len(people_id)):
    y_final.append([people_id[i],y_ran_for[i]])
pd.DataFrame(y_final, columns=[dataset.columns[0],'infect_prob']).to_csv('Pred_1_ranfor.csv')



'''SVR'''

# Fitting the Regression Model to the dataset

#Fitting SVR
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
Xtest = sc.transform(Xtest)

scy= StandardScaler()
y_train = scy.fit_transform(y_train)
y_test = scy.transform(y_test)

model_name = 'SVR'
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
#regressor.fit(X_train,y_train)
regressor.fit(X_train ,y_train )
# Predicting a new result
y_pred = regressor.predict(X_test)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(scy.inverse_transform(y_test),scy.inverse_transform(y_pred))
mse_errors.append([model_name,mse])
plt.scatter(scy.inverse_transform(y_test),scy.inverse_transform(y_pred), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Real')
plt.ylabel('Predicted')
plt.show()

y_svr = regressor.predict(Xtest)
y_final = []
for i in range(len(people_id)):
    y_final.append([people_id[i],y_svr[i]])
pd.DataFrame(y_final, columns=[dataset.columns[0],'infect_prob']).to_csv('Pred_1_svr.csv')



'''Multi Linear Regressiom'''
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

model_name = 'Multiple Linear Regression'
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)


y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test,y_pred)
mse_errors.append([model_name,mse])


plt.scatter(y_test,y_pred)
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Real')
plt.ylabel('Predicted')
plt.show()

y_LinReg = regressor.predict(Xtest)
'''ANN'''

#Fitting ann
#Splitting Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
Xtest = sc.transform(Xtest)


scy= StandardScaler()
y_train = scy.fit_transform(y_train)
y_test = scy.transform(y_test)

model_name='ANN'
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 17, init = 'uniform', activation = 'relu', input_dim = 34))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 17, init = 'uniform', activation = 'relu'))
# Adding the second hidden layer
classifier.add(Dense(output_dim = 17, init = 'uniform', activation = 'relu'))
# Adding the second hidden layer
classifier.add(Dense(output_dim = 17, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 200, nb_epoch = 50)

# Part 3 - Making the predictions and evaluating the model
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = scy.inverse_transform(y_pred)
mse = mean_squared_error(scy.inverse_transform(y_test),y_pred)
mse_errors.append([model_name,mse])

plt.scatter(scy.inverse_transform(y_test),y_pred, color = 'blue')
plt.title('Truth or Bluff (ANN Model)')
plt.xlabel('Real')
plt.ylabel('Predicted')
plt.show()

'''Conclusion'''
'''
mse_errors:-

[['Random Forest', 81.41187910796387],
 ['SVR', 87.70487367605601],
 ['ANN', 98.29827112299039],
 ['Multiple Linear Regression', 87.43791803735338]]
So far we have got the best result for Random Forest Regressor with n_estimators = 1000
Now let us try to optimize it a bit further

'''
#Splitting Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


#Optimizin Random Forest Regressor
model_name = 'Random Forest(2000), min_leaf(5) min_samples_split(5)'
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 2000,
                                  min_samples_leaf=5,
                                  random_state = 45)
regressor.fit(X_train,y_train)


# Predicting a new result
y_pred = regressor.predict(X_test)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,y_pred)
mse_errors.append([model_name,mse])
plt.scatter(y_test,y_pred, color = 'blue')
plt.title('infect_prob_27th_march (Random Forest Regression Model)')
plt.xlabel('Real')
plt.ylabel('Predicted')
plt.savefig('infect_prob_27th_march')
plt.show()

#Predicting on Test_datasset and saving
y_ran_for = regressor.predict(Xtest)

y_final = []
for i in range(len(people_id)):
    y_final.append([people_id[i],y_ran_for[i]])
pd.DataFrame(y_final, columns=[dataset.columns[0],'infect_prob']).to_csv('Pred_2_final_27_march.csv')
