import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
from category_encoders.one_hot import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,cross_val_score

##Loading and Imputing the Data
df = pd.read_csv('train.csv')

# print(df.head())
# print(df.isnull().values.any())
nul_col = df.columns[df.isnull().any()].tolist()
for col in nul_col:
     if df[col].dtype == object:
         imp = SimpleImputer(strategy='most_frequent')
         df[col] = imp.fit_transform(df[[col]])
     else:
         imp = SimpleImputer(strategy='mean')
         df[col] = imp.fit_transform(df[[col]])


## Analysing the Data
# my_report = sweetviz.analyze([df,'Train'], target_feat= 'G3')
# my_report.show_html()


## Scaling and Encoding the data
for colum in df.columns:
    if df[colum].dtype == object:
        # print(colum , df[colum].unique().tolist())
        df[colum] = OneHotEncoder().fit_transform(df[colum])

columns = df.columns
df = MinMaxScaler().fit_transform(df)
df = pd.DataFrame(df,columns=columns)


## Finding the Correlations between Features
# sns.heatmap(df.corr(), fmt = '.1f',annot = True)
# plt.show()

correlations = df.corr()['SalePrice'].drop('SalePrice')
# print(correlations)
# print(correlations.quantile(.25))
# print(correlations.quantile(.75))
# print(correlations.quantile(.50))


## Choosing the best threshold for improving the model
def get_features(correlation_threshold):
    abs_corrs = correlations.abs()
    high_correlations = abs_corrs[abs_corrs > correlation_threshold].index.values.tolist()
    return high_correlations

# thresh = []
# scores = []
# for i in np.arange(start = 0.08,stop = 0.29,step = 0.01):
#     features = get_features(i)
#     thresh.append((i))
#     X = df[features]
#     Y = df.SalePrice
#
#     x_train, x_valid, y_train, y_valid = train_test_split(X, Y, random_state=4)
#     regressor = RandomForestRegressor(n_estimators= 100,random_state=6)
#     regressor.fit(x_train, y_train)
#     score = regressor.score(x_valid, y_valid)
#     scores.append(score)
#
# plt.plot(thresh,scores)
# plt.xlabel('thrshold_values')
# plt.ylabel('scores')
# plt.show()


## Final Threshold with greatest Score
features = get_features(0.22)
# print('number of features required to predict the Saleprice {}'.format(len(features)))
# print(features)
X = df[features]
Y = df.SalePrice


## Spliting, Fiting, Cross_Validating Model
x_train, x_valid, y_train, y_valid = train_test_split(X, Y, random_state=4)
regressor = RandomForestRegressor(n_estimators= 100,random_state=6)
regressor.fit(x_train, y_train)
print("R2 Score of the regression model is :-",regressor.score(x_valid, y_valid))
print(regressor.score(x_valid,y_valid))

accuracies = cross_val_score(regressor,X,Y,scoring = 'r2',cv = 10)
print(accuracies.mean())
print(accuracies.std())


## Working on the test Dataset
test_df = pd.read_csv('test.csv')
# print(test_df.head())
# print(test_df.isnull().values.any())
nul_col = test_df.columns[test_df.isnull().any()].tolist()
for col in nul_col:
     if test_df[col].dtype == object:
         imp = SimpleImputer(strategy='most_frequent')
         test_df[col] = imp.fit_transform(test_df[[col]])
     else:
         imp = SimpleImputer(strategy='mean')
         test_df[col] = imp.fit_transform(test_df[[col]])


## Scaling and Encoding the test dataset
for colum in test_df.columns:
    if test_df[colum].dtype == object:
        # print(colum , test_df[colum].unique().tolist())
        test_df[colum] = OneHotEncoder().fit_transform(test_df[colum])

## Predicting and Saving the Prediction into CSV File
test_def_sel = test_df[['MSZoning', 'LotFrontage', 'LotArea', 'LotShape', 'HouseStyle', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'Exterior1st', 'Exterior2nd', 'MasVnrArea', 'ExterQual', 'Foundation', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'TotalBsmtSF', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'BsmtFullBath', 'FullBath', 'HalfBath', 'KitchenQual', 'TotRmsAbvGrd', 'Fireplaces', 'GarageType', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'SaleType']]
predictions = regressor.predict(test_def_sel)
# print(predictions)
ids = test_df['Id'].tolist()
data = {'Id':ids,'SalePrice':predictions}
Output = pd.DataFrame(data).to_csv(r'C:\Users\Vibhu Gupta\PycharmProjects\Machine Learning Projects\Kaggle House Prediction comp\CSV File of predictions',index = False)
# print(Output)