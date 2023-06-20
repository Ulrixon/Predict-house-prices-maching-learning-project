#%%

from tensorflow import keras
from tensorflow.keras import layers

from keras.constraints import maxnorm

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import metrics
import joblib
from joblib import parallel_backend
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

#%%
df_old = pd.read_csv(r"/Users/ryan/Documents/vscode(py)/train.csv", index_col="Id")
data_test = pd.read_csv(r"/Users/ryan/Documents/vscode(py)/test.csv", index_col="Id")
SalePrice = df_old.iloc[:, -1]
df_old = df_old.drop(["SalePrice"], axis=1)
df = pd.concat([df_old, data_test], axis=0)

df = df.drop(["PoolQC", "Alley", "MiscFeature", "Fence", "FireplaceQu",], axis=1)

# 找出剩下有缺失值的變數，並將其顯示出來
for i in df.columns:
    if df[i].isnull().sum() != 0:
        print(i, df[i].isnull().sum(), df[i].dtype)

# 建立數值變數跟類別變數的集合
num_col = df._get_numeric_data().columns.tolist()  # 數值變數有36個
cat_col = set(df.columns) - set(num_col)  # 類別變數有38個(74-36=38)
cat_col = list(cat_col)  # 把set轉成list

# 有數值的變數用平均填補缺失值，代表類型的變數用眾數填補缺失值
for col in num_col:
    df[col].fillna(df[col].mean(), inplace=True)
for col in cat_col:
    df[col].fillna(df[col].mode()[0], inplace=True)


le = preprocessing.LabelEncoder()
for i in range(len(df.iloc[1])):  # len(df.iloc[1])來呈現regressor數量
    if type(df.iloc[1, i]) == str:  # 判別是否regressor是類別變數
        df.iloc[:, i] = le.fit_transform(df.iloc[:, i])  # 若是類別變數，則將那格改成labelencoder方式表示

#%%
# 標準化
# Standardizing the features(數值變數)
df[num_col] = StandardScaler().fit_transform(df[num_col])

# pca(數值變數)
pca = PCA(n_components=36)
principalComponents = pca.fit_transform(df[num_col])

# 計算累積貢獻比率
print(np.cumsum(pca.explained_variance_ratio_))
# 決定挑20個，因為此時累積貢獻比率達到0.85

# 畫累積貢獻比率圖
plt.figure()
plt.plot(np.arange(1, 37), np.cumsum(pca.explained_variance_ratio_), linewidth=2)
plt.xlabel("components_n", fontsize=16)
plt.ylabel("explained_variance_", fontsize=16)
plt.show()

# n_components=20
pca = PCA(n_components=20)
principalComponents = pca.fit_transform(df[num_col])
principalComponents = pd.DataFrame(principalComponents)

# 畫出前3個pca
fig1, ax = plt.subplots(3, 1, figsize=(9, 4), dpi=100)
for jj in range(3):
    ax[jj].plot(principalComponents.iloc[:, jj], lw=2.5, label="PC" + str(jj + 1))
    ax[jj].legend(loc="upper left")
fig1.tight_layout(pad=1.5)

# 讓principalComponents的index從1開始
principalComponents.index = principalComponents.index + 1
# 將經過pca的數值變數跟數值變數合併
df = pd.concat([principalComponents, df[cat_col]], axis=1)

#%%
# data split
X = df.iloc[:1460, :]  # train set的regressor
y = SalePrice  # train set的predictor
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# %%
# optuna+stacking
def objective(trial, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test):

    params_xgb = {
        "max_depth": trial.suggest_int("max_depth", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 2000, 4000, 100),
        "eta": trial.suggest_float("eta", 1e-8, 1.0, log=True),
        # "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        # "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
        # "grow_policy": trial.suggest_categorical(
        #    "grow_policy", ["depthwise", "lossguide"]
        # ),
        # "hidden_layer_sizes": trial.suggest_int()
    }
    params_rf = {
        # "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "max_depth": trial.suggest_int("max_depth", 4, 50),
        # "min_samples_split": trial.suggest_int("min_samples_split", 1, 150),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 60),
    }

    params_mlp = {
        "learning_rate_init": trial.suggest_float(
            "learning_rate_init ", 0.0001, 0.1, step=0.005
        ),
        "first_layer_neurons": trial.suggest_int("first_layer_neurons", 10, 100),
        "second_layer_neurons": trial.suggest_int("second_layer_neurons", 10, 100),
        "activation": trial.suggest_categorical(
            "activation", ["identity", "tanh", "relu"]
        ),
    }

    # svr para

    # svr para
    gamma = trial.suggest_loguniform("svr_gamma", 1e-5, 1e5)

    # cat para
    param_cat = {
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
        "depth": trial.suggest_int("depth", 1, 12),
    }

    # if param_cat["bootstrap_type"] == "Bayesian":
    #    param_cat["bagging_temperature"] = trial.suggest_float(
    #        "bagging_temperature", 0, 10
    #    )
    # elif param_cat["bootstrap_type"] == "Bernoulli":
    #    param_cat["subsample"] = trial.suggest_float("subsample", 0.1, 1)

    estimators = [
        ("rf", RandomForestRegressor(**params_rf)),
        ("xgb", XGBRegressor(**params_xgb)),
        ("svr", svm.SVR(gamma=gamma)),
        ("lasso", Lasso(alpha=1)),
        ("cat", CatBoostRegressor(**param_cat)),
    ]

    clf = StackingRegressor(
        estimators=estimators,
        final_estimator=MLPRegressor(
            hidden_layer_sizes=(
                params_mlp["first_layer_neurons"],
                params_mlp["second_layer_neurons"],
            ),
            learning_rate_init=params_mlp["learning_rate_init"],
            activation=params_mlp["activation"],
            max_iter=1000,
        ),
    )

    # params = {}

    stack1 = clf.fit(X_train, y_train)
    # 使用訓練資料預測
    # train_pred = clf.predict(X_train)
    # 使用測試資料預測
    test_pred = stack1.predict(X_test)

    # mse = metrics.mean_squared_error(y_train, train_pred)
    # print('訓練集 MSE: ', mse)
    mse = metrics.mean_squared_error(y_test, test_pred)
    print("測試集 MSE: ", mse)
    return mse


# Creating Optuna object and defining its parameters
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=40)

# Showing optimization results
print("Number of finished trials:", len(study.trials))
print("Best trial parameters:", study.best_trial.params)
print("Best score:", study.best_value)
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
#%%
# grid stacking
light_params = {
    "task": "train",
    "objective": "regression",
    "num_leaves": 10,
    "learnnig_rage": 0.05,
}

estimators = [
    ("rf", RandomForestRegressor(n_estimators=5)),
    ("xgb", XGBRegressor()),
    ("svr", svm.SVR()),
    ("lasso", Lasso(alpha=1.0)),
    ("cat", CatBoostRegressor()),
]

clf = StackingRegressor(
    estimators=estimators,
    final_estimator=MLPRegressor(
        activation="relu",
        alpha=0.1,
        hidden_layer_sizes=(20, 20),
        learning_rate="constant",
        max_iter=2000,
    ),
)

params = {"rf__n_estimators": [1, 100]}
# params = {}
grid = GridSearchCV(estimator=clf, param_grid=params, cv=5)
grid.fit(X_train, y_train)
# 使用訓練資料預測
train_pred = grid.predict(X_train)
# 使用測試資料預測
test_pred = grid.predict(X_test)

mse = metrics.mean_squared_error(y_train, train_pred)
print("訓練集 MSE: ", mse)
mse = metrics.mean_squared_error(y_test, test_pred)
print("測試集 MSE: ", mse)

