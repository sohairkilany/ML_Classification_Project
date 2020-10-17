## Load Import necessary dependencies
import numpy as np      # linear algebra
import os               # accessing directory structure
import pandas as pd     # data processing, CSV file I/O
import matplotlib.pyplot as plt         # plotting
import seaborn as sns
from sklearn import preprocessing

## Load and Read DataSets
df = pd.read_csv('train.csv', sep=',', na_values=['N/A', 'no', '?'])
## Return the first n rows.
print(df.head(10))           # n= 10
## method for prints information about a DataFrame including the index dtype and columns, non-null values and memory usage
df.info()      # rows = 318438 , columns = 18
## To Visualize Data
##pairplot will plot pairwise relationships across an entire dataframe (for the numerical columns)
##and supports a color hue argument (for categorical columns)
# sns.pairplot(df)
# plt.savefig('visualize.jpg')
# plt.show()
# sns.pairplot(df, hue='Stay', height=3, aspect=1.3)  #Use hue to show graph based on the hue category values
# plt.savefig('visualize_hue_stay.jpg')
# plt.show()

#### Feature Transformations
## Check Missing Data
### number of missing data in City_Code_Patient column = 4532,   number of missing data in Bed Grade = 113 ,
###missing data is less, so drop rows which contain missing data
print(df.isnull().sum())
print(df.describe())
print(df['Age'].value_counts())
## Work with Missing Data
## Drop Missing Data
df = df.dropna(axis=0)  # drop rows from a data set containing missing values
df.info()
## Check Missing Data
print(df.isnull().sum())

columns =['case_id', 'Hospital_type_code', 'Hospital_region_code', 'Ward_Facility_Code', 'patientid', 'City_Code_Patient']
df = df.drop(columns, axis=1)
df.info()
## # Converting float64 to int type
df['Bed Grade'] = df['Bed Grade'].astype(np.int64)
df['Admission_Deposit'] = df['Admission_Deposit'].astype(np.int64)

df.info()
## Work with Categorical Data  ## columns [Department, Ward_Type, Type of Admission, Severity of Illness, Age, Stay ]

df = pd.get_dummies(df, columns=['Department', 'Ward_Type', 'Type of Admission', 'Severity of Illness', 'Age'], drop_first=True)
print(df.columns)
df.info()                   # (total 29 columns), 313793 entries (rows)
le = preprocessing.LabelEncoder()       # to convert Y from categorical to label encoder
df['Stay'] = le.fit_transform(df['Stay'])
df['Stay'] = df['Stay'].astype(int)
print(list(le.classes_)) #['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100', 'More than 100 Days']
#print(list(le.inverse_transform([0, 1, 2,3, 4, 5, 6, 7, 8, 9, 10])))

## Detect and Handle Outliers
columns =['Hospital_code','City_Code_Hospital', 'Available Extra Rooms in Hospital', 'Bed Grade', 'Visitors with Patient', 'Admission_Deposit']
# for col in columns:     # to show outliers
#     sns.boxplot(x=col, data=df)
#     sns.stripplot(x=col, data=df, color="#474646")
#     plt.show()

from datasist.structdata import detect_outliers
outliers_indices = detect_outliers(df, 0, columns)
print(len(outliers_indices))
# handle outliers
df.drop(outliers_indices, inplace=True)
df.info()

### Deal with Imbalanced classes  ## Stay column
print(df['Stay'].value_counts())
from sklearn.model_selection import train_test_split
x = df.drop('Stay', axis=1)
y = df['Stay']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=22)

#####################
print(x_train.shape)
print(x_test.iloc[50])
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=22)
# make smote in only training set
X_train, Y_train = smote.fit_sample(x_train, y_train)

from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
scaling = StandardScaler()
X_train = scaling.fit_transform(X_train)
X_test = scaling.transform(x_test)
print(X_train.shape)
print(X_test.shape)


# #### Train model
from sklearn.model_selection import cross_validate, cross_val_predict, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# # first model  logistic regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs')

## in case used cross Validation
# cv_result = cross_validate(model, X_train, Y_train, cv=10, return_train_score=True)
# print(cv_result)
# print(cv_result['test_score'].mean())
# Y_predict = cross_val_predict(model, X_test, y_test, cv=10)

# ################## to evaluate Model
model.fit(X_train, Y_train)
Y_predict = model.predict(x_test)
print(accuracy_score(Y_predict, y_test))
# import joblib
# joblib.dump(model, 'model.pkl')

# #to load pretrained model and test for one Example
# import joblib
# model = joblib.load('model.pkl')
#
# print(model.predict([[8, 3, 4, 2,2, 5451,0,1,0,0 ,0,1,0,0,0, 0,1,0,1,0,1,0,0,0,0,0,0,0]])[0])
# print(model.predict([X_test[50]])[0])