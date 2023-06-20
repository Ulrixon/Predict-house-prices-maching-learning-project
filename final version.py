# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 17:57:25 2022

@author: tp65k
"""



#%%
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from skfeature.function.similarity_based import fisher_score
from matplotlib.pyplot import figure
from catboost import CatBoostRegressor
import optuna

%matplotlib inline

#%%
#載入資料，並把index那行改成id
df_old= pd.read_csv(r'/Users/ryan/Documents/vscode(py)/train.csv',index_col='Id')
data_test = pd.read_csv(r'/Users/ryan/Documents/vscode(py)/test.csv',index_col='Id')
SalePrice = df_old.iloc[:,-1]
df_old = df_old.drop(['SalePrice'],axis=1)
df=pd.concat([df_old,data_test],axis=0) # concat垂直合併

df.info()
#資料未達8成(2335筆)的變數為:'Alley','FireplaceQu','PoolQC','Fence','MiscFeature'
#2919*0.8=2335.2

#刪掉這些變數(5個，剩74個)
df = df.drop(['PoolQC','Alley','MiscFeature','Fence','FireplaceQu',],axis=1)

#找出剩下有缺失值的變數，並將其顯示出來
for i in df.columns:
    if (df[i].isnull().sum()!=0):
        print(i,df[i].isnull().sum(),df[i].dtype)

#建立數值變數跟類別變數的集合
num_col=df._get_numeric_data().columns.tolist() #數值變數有36個
cat_col=set(df.columns)-set(num_col) #類別變數有38個(74-36=38)
cat_col=list(cat_col) #把set轉成list

#有數值的變數用平均填補缺失值，代表類型的變數用眾數填補缺失值
for col in num_col:
    df[col].fillna(df[col].mean(),inplace=True)
for col in cat_col:
    df[col].fillna(df[col].mode()[0],inplace=True)

#%% 
#測出所有類型變數的資料分布
for i in list(cat_col):
    print(df[i].value_counts()) 

#自己找某實現值比例超過80%(即多於2919*0.8=2335筆)的變數，共20個 
'PavedDrive,ExterCond,BldgType,RoofMatl,LandSlope,BsmtCond,Heating,Condition2'  
'LandContour,Functional,Street,BsmtFinType2,Utilities,Condition1,GarageQual,CentralAir'  
'GarageCond,Electrical,SaleCondition,SaleType'


#2
#將實現值比例超過80%的變數刪除(20個,剩54個)
df = df.drop(["PavedDrive","ExterCond","BldgType","RoofMatl","LandSlope","BsmtCond","Heating",
              "Condition2","LandContour","Functional","Street","BsmtFinType2","Utilities",
              "Condition1","GarageQual","CentralAir","GarageCond","Electrical","SaleCondition",
              "SaleType"],axis=1)

del_col =["PavedDrive","ExterCond","BldgType","RoofMatl","LandSlope","BsmtCond","Heating",
              "Condition2","LandContour","Functional","Street","BsmtFinType2","Utilities",
              "Condition1","GarageQual","CentralAir","GarageCond","Electrical","SaleCondition",
              "SaleType"]

cat_col = set(cat_col) - set(del_col)
cat_col=list(cat_col) #把set轉成list

#%%
#只看train set的相關性，test set沒房價沒辦法看相關性
df_prime =pd.concat([df_old,SalePrice],axis=1) #concat水平合併
corrmat = df_prime.corr()

#畫出與房價相關係數大於0.5的變數跟房價的heatmap
top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.5] #發現有10個變數與房價的相關性大於0.5
plt.figure(figsize=(9,9))
gragh = sns.heatmap(df_prime[top_corr_features].corr(),annot=True,cmap="Blues") 
#annot=True -> write the data value in each cell
'OverallQual,YearBuilt,YearRemodAdd,TotalBsmtSF,1stFlrSF'
'GrLivArea,FullBath,TotRmsAbvGrd,GarageCars,GarageArea'  

#顯示出所有變數與房價的相關係數
var = df_prime[df_prime.columns[1:]].corr()['SalePrice'][:]
var.sort_values(ascending=False)

#%%   
#處理類別變數(label encoder) (仍是74個變數)
le = preprocessing.LabelEncoder()
for i in range(len(df.iloc[1])):  #len(df.iloc[1])來呈現regressor數量
    if (type(df.iloc[1,i])== str ): #判別是否regressor是類別變數
        df.iloc[:,i]=le.fit_transform(df.iloc[:,i]) #若是類別變數，則將那格改成labelencoder方式表示
        
#%%
#標準化
# Standardizing the features(數值變數)
df[num_col]=StandardScaler().fit_transform(df[num_col])

#pca(數值變數)
pca = PCA(n_components=36)
principalComponents = pca.fit_transform(df[num_col])

#計算累積貢獻比率
print(np.cumsum(pca.explained_variance_ratio_)) 
#決定挑20個，因為此時累積貢獻比率達到0.85

#畫累積貢獻比率圖
plt.figure()
plt.plot(np.arange(1,37),np.cumsum(pca.explained_variance_ratio_),linewidth=2)
plt.xlabel('components_n', fontsize=16)
plt.ylabel('explained_variance_', fontsize=16)
plt.show()

#n_components=20
pca = PCA(n_components=20)
principalComponents = pca.fit_transform(df[num_col])
principalComponents = pd.DataFrame(principalComponents)

#畫出前3個pca
fig1 , ax = plt.subplots(3,1,figsize=(9,4),dpi=100)
for jj in range(3):
    ax[jj].plot(principalComponents.iloc[:,jj], lw=2.5, label='PC'+str(jj+1))
    ax[jj].legend(loc='upper left')
fig1.tight_layout(pad=1.5)

#讓principalComponents的index從1開始
principalComponents.index = principalComponents.index + 1
#將經過pca的數值變數跟數值變數合併
df = pd.concat([principalComponents,df[cat_col]],axis=1)

#%%
#data split
X= df.iloc[:1460,:].to_numpy() #train set的regressor
y= SalePrice #train set的predictor
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

#%%
ranks = fisher_score.fisher_score(X.to_numpy(),y.to_numpy())

#%%
#grid stacking
light_params = {
    'task': 'train',
    'objective': 'regression',
    'num_leaves': 10,
    'learnnig_rage': 0.05}

estimators = [
    ('rf', RandomForestRegressor(n_estimators=5)),
    ('xgb', XGBRegressor()),
    ('svr', svm.SVR()),
    ('lasso',Lasso(alpha=1.0)),
    ('cat',CatBoostRegressor())
    ]

clf = StackingRegressor(
            estimators=estimators, final_estimator= MLPRegressor(activation = "relu", alpha = 0.1, hidden_layer_sizes = (20,20),
                                    learning_rate = "constant",solver= "adam", max_iter = 5000)
)

params = {'rf__n_estimators': [1,100]}
#params = {}
grid = GridSearchCV(estimator=clf, param_grid=params, cv=5)
grid.fit(X_train, y_train)
# 使用訓練資料預測
train_pred = grid.predict(X_train)
# 使用測試資料預測
test_pred = grid.predict(X_test)

mse = metrics.mean_squared_error(y_train, train_pred)
print('訓練集 MSE: ', mse)
mse = metrics.mean_squared_error(y_test, test_pred)
print('測試集 MSE: ', mse)


#%%
#xgboost
# 建立 XGBRegressor 模型
xgboostModel = XGBRegressor()
params = {'n_estimators':[1,1000]}

# 使用訓練資料訓練模型
grid = GridSearchCV(estimator=xgboostModel, param_grid=params, cv=5)
grid.fit(X_train, y_train)
# 使用訓練資料預測
train_pred = grid.predict(X_train)
# 使用測試資料預測
test_pred = grid.predict(X_test)

mse = metrics.mean_squared_error(y_train, train_pred)
print('訓練集 MSE: ', mse)
mse = metrics.mean_squared_error(y_test, test_pred)
print('測試集 MSE: ', mse)

#%%
#lasso
lasso = Lasso(max_iter = 10000, normalize = True) #set the maximum number of iterations=10000
coefs = []
alphas = 10**np.linspace(10,-2,100)*0.5

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(scale(X_train), y_train)
    coefs.append(lasso.coef_)

lassocv = LassoCV(alphas = None, cv = 10, max_iter = 100000, normalize = True)
lassocv.fit(X_train, y_train)

lasso.set_params(alpha=lassocv.alpha_)
lasso.fit(X_train, y_train)
mse = metrics.mean_squared_error(y_train, lasso.predict(X_train))
print('訓練集 MSE: ', mse)
mse = metrics.mean_squared_error(y_test, lasso.predict(X_test))
print('測試集 MSE: ', mse)

#%%
#分出test set資料
data_test = df.iloc[1460:,:].to_numpy()

#stacking
y_stacking_hat = grid.predict(data_test)
prediction = pd.DataFrame(y_stacking_hat)
sub = pd.read_csv(r'/Users/ryan/Downloads/sample_submission.csv')
dataset = pd.concat([sub['Id'],prediction],axis=1)
dataset.columns=['Id','SalePrice']
dataset.to_csv(path_or_buf='/Users/ryan/Downloads/sample_submission21.csv',index=False)
#%%
#xgboost
y_boost_hat = grid.predict(data_test)
prediction = pd.DataFrame(y_boost_hat)
sub = pd.read_csv(r'C:/Users/tp65k/Desktop/sample_submission.csv')
dataset = pd.concat([sub['Id'],prediction],axis=1)
dataset.columns=['Id','SalePrice']
dataset.to_csv('sample_submission22.csv',index=False)

#lasso q
y_lasso_hat = lasso.predict(data_test)
prediction = pd.DataFrame(y_lasso_hat)
sub = pd.read_csv(r'C:/Users/tp65k/Desktop/sample_submission.csv')
dataset = pd.concat([sub['Id'],prediction],axis=1)
dataset.columns=['Id','SalePrice']
dataset.to_csv('sample_submission23.csv',index=False)




















# %%
#optuna+stacking
def objective(trial, X=X, y=y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    params_xgb = {
        
        'max_depth': trial.suggest_int('max_depth', 6, 15),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000, 100),
        'eta': trial.suggest_float("eta", 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'gamma': trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 2, 10),
        'grow_policy': trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        #"hidden_layer_sizes": trial.suggest_int()
    }
    params_rf={
        
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'max_depth': trial.suggest_int('max_depth', 4, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 1, 150),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 60),
        
    }
    
    params_mlp = {
        'learning_rate_init': trial.suggest_float('learning_rate_init ', 0.0001, 0.1, step=0.005),
        'first_layer_neurons': trial.suggest_int('first_layer_neurons', 10, 100, step=10),
        'second_layer_neurons': trial.suggest_int('second_layer_neurons', 10, 100, step=10),
        'activation': trial.suggest_categorical('activation', ['identity', 'tanh', 'relu']),
    }
    
    estimators = [
    ('rf', RandomForestRegressor(**params_rf)),
    ('xgb', XGBRegressor(**params_xgb)),
    ('svr', svm.SVR()),
    ('lasso',Lasso(alpha=1.0)),
    ('cat',CatBoostRegressor())
    ]

    
    
    clf = StackingRegressor(
            estimators=estimators, final_estimator= MLPRegressor(hidden_layer_sizes=(params_mlp['first_layer_neurons'], params_mlp['second_layer_neurons']),
        learning_rate_init=params_mlp['learning_rate_init'],
        activation=params_mlp['activation'],
        
        max_iter=1000)
)

    
    #params = {}

    stack1=clf.fit(X_train,y_train)
    # 使用訓練資料預測
    #train_pred = clf.predict(X_train)
    # 使用測試資料預測
    test_pred = stack1.predict(X_test)

    #mse = metrics.mean_squared_error(y_train, train_pred)
    #print('訓練集 MSE: ', mse)
    mse = metrics.mean_squared_error(y_test, test_pred)
    print('測試集 MSE: ', mse)
    return mse
# Creating Optuna object and defining its parameters
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials = 10)

# Showing optimization results
print('Number of finished trials:', len(study.trials))
print('Best trial parameters:', study.best_trial.params)
print('Best score:', study.best_value)
# %%
from optuna.visualization import plot_optimization_history

plotly_config = {"staticPlot": True}

fig = plot_optimization_history(study)
fig.show(config=plotly_config)
#%%
from optuna.visualization import plot_param_importances

fig = plot_param_importances(study)
fig.show(config=plotly_config)
# %%
