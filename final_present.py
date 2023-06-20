#%%
from ast import increment_lineno
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from xgboost import XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from skfeature.function.similarity_based import fisher_score
from matplotlib.pyplot import figure

%matplotlib inline


#%%
#載入資料，並把index那行改成id
df = pd.read_csv(r'/Users/ryan/Documents/vscode(py)/train.csv',index_col='Id')

df.info()
#發現有很多缺失值的變數為:'Alley','FireplaceQu','PoolQC','Fence','MiscFeature' (<1300筆)

#刪掉這些變數
df = df.drop(['PoolQC','Alley','MiscFeature','Fence','FireplaceQu'],axis=1)

#找出有缺失值但不多(<100)的變數，並將其顯示出來
for i in df.columns:
    if (df[i].isnull().sum()!=0):
        print(i,df[i].isnull().sum(),df[i].dtype)


#%%        
#有數值的變數用mean填補缺失值，代表類型的變數用mode填補缺失值
df['LotFrontage'].fillna(df['LotFrontage'].mean(),inplace=True)
df['MasVnrArea'].fillna(df['MasVnrArea'].mean(),inplace=True)
df['MasVnrType'].fillna(df['MasVnrType'].mode()[0],inplace=True)
df['BsmtQual'].fillna(df['BsmtQual'].mode()[0],inplace=True)
df['BsmtCond'].fillna(df['BsmtCond'].mode()[0],inplace=True)
df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0],inplace=True)
df['BsmtFinType1'].fillna(df['BsmtFinType1'].mode()[0],inplace=True)
df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0],inplace=True)
df['Electrical'].fillna(df['Electrical'].mode()[0],inplace=True)
df['GarageType'].fillna(df['GarageType'].mode()[0],inplace=True)
df['GarageYrBlt'].fillna(df['GarageYrBlt'].mean(),inplace=True)
df['GarageFinish'].fillna(df['GarageFinish'].mode()[0],inplace=True)
df['GarageQual'].fillna(df['GarageQual'].mode()[0],inplace=True)
df['GarageCond'].fillna(df['GarageCond'].mode()[0],inplace=True)
#將資料變成一個沒有缺失值的dataframe
le = preprocessing.LabelEncoder()
for i in range(len(df.iloc[1])):
    if (type(df.iloc[0,i])== str ):
        df.iloc[:,i]=le.fit_transform(df.iloc[:,i])
        
       



#%%
#data split
X= df.iloc[:,:-1]
y= df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
print('Training data shape:', X_train.shape)
print('Testing data shape:', X_test.shape)

#%%
ranks = fisher_score.fisher_score(X.to_numpy(),y.to_numpy())
fisherX_train=X_train.iloc[:,ranks >20]
fisherX_test= X_test.iloc[:,ranks >20]
#%%

feat_importance= pd.DataFrame(ranks,X.columns)



figure(figsize=(20, 10), dpi=100)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.xticks(rotation = 90)
plt.bar(feat_importance.index,feat_importance.iloc[:,0],color='teal')
plt.figure

#%%
cor = df.corr()
figure(figsize=(100, 100), dpi=100)
sns.set(font_scale=1.4)
sns.heatmap(cor,annot=True)



#%%
#grid stacking
light_params = {
    'task': 'train', 
    #'boosting': 'gbdt',
    'objective': 'regression',
    'num_leaves': 10,
    'learnnig_rage': 0.05,
    #'metric': {'l2','l1'},
    #'verbose': -1
}
 
estimators = [
    ('rf', RandomForestRegressor(n_estimators=5)),
    ('xgb', XGBRegressor()),
    ('svr', svm.SVR()),
    #('light',lgb.LGBMRegressor())
    #('knn', KNeighborsRegressor()),
    #('dt', DecisionTreeRegressor(random_state = 42))
]
clf = StackingRegressor(
            estimators=estimators, final_estimator= MLPRegressor(activation = "relu", alpha = 0.1, hidden_layer_sizes = (20,20),
                                    learning_rate = "constant", max_iter = 2000, random_state = 1000)
)
params = {'rf__n_estimators': [1,100],
              #'light__num_leaves': [31, 127],
            #'light__reg_alpha': [0.1, 0.5],
            #'light__min_data_in_leaf': [30, 50, 100, 300, 400],
            #'light__lambda_l1': [0, 1, 1.5],
            #'light__lambda_l2': [0, 1]
          #'xgb__n_estimators':[1,1000],
          #'xgb__max_depth': [1,1000]
          }
grid = GridSearchCV(estimator=clf, param_grid=params, cv=5)
grid.fit(X_train, y_train)
test_pred = grid.predict(X_test)
train_pred = grid.predict(X_train)
mse = metrics.mean_squared_error(y_train, train_pred)
print('訓練集 MSE: ', mse)
mse = metrics.mean_squared_error(y_test, test_pred)
print('測試集 MSE: ', mse)


#%%
#xgboost
# 建立 XGBRegressor 模型
xgboostModel = XGBRegressor()
params = {
          'n_estimators':[1,1000],
          #'max_depth': [1,1000]
          #"min_child_weight": [ 1, 3, 5, 7],
          #"gamma":[ 0.0, 0.1, 0.2],
          #"colsample_bytree":[ 0.3, 0.4],
          }
# 使用訓練資料訓練模型
grid = GridSearchCV(estimator=xgboostModel, param_grid=params, cv=5)
grid.fit(X_train, y_train)
# 使用訓練資料預測
train_pred = grid.predict(X_train)

mse = metrics.mean_squared_error(y_train, train_pred)
print('訓練集 MSE: ', mse)
# 測試集 MSE
test_pred = grid.predict(X_test)
mse = metrics.mean_squared_error(y_test, test_pred)
print('測試集 MSE: ', mse)
# %%
