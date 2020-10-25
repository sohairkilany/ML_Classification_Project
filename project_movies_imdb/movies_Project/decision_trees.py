## Load Import necessary dependencies
import numpy as np      # linear algebra
import os               # accessing directory structure
import pandas as pd     # data processing, CSV file I/O
import matplotlib.pyplot as plt         # plotting
import seaborn as sns
from sklearn import preprocessing

## Load and Read DataSets
df = pd.read_csv('MACHINE_LEARNING_FINAL.csv', sep=';', na_values=['N/A', 'no', '?'])
## Return the first n rows.
print(df.head(10))           # n= 10
## method for prints information about a DataFrame including the index dtype and columns, non-null values and memory usage
df.info()      # rows = 3750 , columns = 37
## To Visualize Data
##pairplot will plot pairwise relationships across an entire dataframe (for the numerical columns)
##and supports a color hue argument (for categorical columns)
# sns.pairplot(df)
# plt.savefig('visualize.jpg')
# plt.show()
# sns.pairplot(df, hue='ratingInteger', height=3, aspect=1.3)  #Use hue to show graph based on the hue category values
# plt.savefig('visualize_hue_stay.jpg')
# plt.show()

#### Feature Transformations
## Check Missing Data
### dataset not contain missing data
print(df.isnull().sum())
print(df.describe())
le = preprocessing.LabelEncoder()       # to convert Y from categorical to label encoder
df['ratingInteger'] = le.fit_transform(df['ratingInteger'])
df['ratingInteger'] = df['ratingInteger'].astype(int)
print(list(le.classes_))   ## [2, 3, 4, 5, 6, 7, 8, 9]
## Detect and Handle Outliers
columns =['year', 'lifetime_gross', 'ratingCount', 'duration', 'nrOfWins', 'nrOfNominations', 'nrOfNewsArticles', 'nrOfUserReviews', 'nrOfGenre']
# for col in columns:     # to show outliers
#     sns.boxplot(x=col, data=df)
#     sns.stripplot(x=col, data=df, color="#474646")
#     plt.show()

from datasist.structdata import detect_outliers
outliers_indices = detect_outliers(df, 0, columns)
print(len(outliers_indices))   # number of outliers = 1085
# handle outliers
df.drop(outliers_indices, inplace=True)
df.info()

## drop title column
df = df.drop('title', axis=1)
df.info()

### Deal with Imbalanced classes  ## Stay column
print(df['ratingInteger'].value_counts())
from sklearn.model_selection import train_test_split
x = df.drop('ratingInteger', axis=1)
y = df['ratingInteger']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=22)

#####################
print(x_train.shape)
print(y_train.shape)

from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
scaling = StandardScaler()
x_train = scaling.fit_transform(x_train)
x_test = scaling.transform(x_test)
print(x_train.shape)
print(x_test.shape)


#### Train model
from sklearn.model_selection import cross_validate, cross_val_predict, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2)

##################  train && to evaluate Model without cross validation
# model.fit(x_train, y_train)
# Y_predict = model.predict(x_test)
# print(accuracy_score(Y_predict, y_test))
# print(confusion_matrix(y_test, Y_predict))
# print(accuracy_score(y_test, Y_predict))
# print(classification_report(y_test, Y_predict))
# print(model.predict([x_test[50]])[0])
# print(y_test.iloc[50])

################## to evaluate Model in case cross Validation
cv_result = cross_validate(model, x_train, y_train, cv=10, return_train_score=True)
print(cv_result)
print(cv_result['test_score'].mean())
Y_predict = cross_val_predict(model, x_test, y_test, cv=10)
print(accuracy_score(Y_predict, y_test))