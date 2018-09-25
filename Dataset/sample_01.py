# import lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from TestCode.models.missing_data import missing_data as msd
from datetime import datetime



#define missing data
# def missing_data(dataset):
#     total = dataset.isnull().sum().sort_values(ascending=False)
#     percent_1 = dataset.isnull().sum()/dataset.isnull().count()*100
#     percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
#     return pd.concat([total, percent_2], axis=1, keys=['筆數', '%'])


# 相關性分析將dataframe的背景色highlight
# def highlight_highcorr(s):
#     is_high = ((s >= 0.6) & (s < 1))
#     return ['background-color: yellow' if v else '' for v in is_high]


# define style
plt.rcParams['font.family']='SimHei' #顯示中文
plt.rcParams['axes.unicode_minus']=False #正常顯示負號
pd.set_option("display.max_columns",50) #設定pandas最多顯示出50個欄位資訊

# create dataset
df = pd.read_csv('data/A_LVR_LAND_A.csv' , encoding='big5')

# 顯示遺失值
print( msd(df) )

# 資料轉換
df['都市土地使用分區'] = df['都市土地使用分區'].replace({'住':1,'其他':2,'商':3,
                                '工':4,'農':5})

df.loc[:,'總價元'] = df.loc[:,'總價元']/10000 #改成以萬為單位，方便圖表顯示

print(df.loc[0,'交易年月日'])
print(type(df.loc[0,'交易年月日']))


for i in range(len(df)):
    df.loc[i,'交易年'] = round(df.loc[i,'交易年月日']/10000)

# define time
now = datetime.now()
now.year #(年,月,日,時,分,秒,微秒) ps: 微秒(microseconds)=1/1000000秒

for i in range(len(df)):
    df.loc[i,'建築完成年'] = round(df.loc[i,'建築完成年月']/10000)
    df.loc[i,'屋齡'] = now.year - 1911 - round(df.loc[i,'建築完成年月']/10000)


# 相關性分析
corr = df[['總價元','鄉鎮市區','建物型態','都市土地使用分區','土地移轉總面積平方公尺','建物移轉總面積平方公尺',
           '建物現況格局-房','建物現況格局-廳','建物現況格局-衛','車位移轉總面積平方公尺','車位總價元','屋齡','建築完成年','交易年']].corr()


plt.figure(figsize=(10,10))
sns.heatmap(corr, square=True, annot=True)
plt.show()


# 產生散佈圖
df.plot(kind='scatter',title='散佈圖',figsize=(10,6),x='建物現況格局-房',y='總價元',marker='+')
plt.show()


# 平均值計算
# 都市土地使用分區：住、其他、商、工、農
print("市土地使用分區：住、其他、商、工、農 - 平均")
print(df.groupby('都市土地使用分區').mean())


# 去除土地和車位兩種交易資料
# df = df[(df['交易標的']!='土地') & (df['交易標的']!='車位')].reset_index(drop=True)


# 各鄉鎮資料筆數 - 長條圖
ax = df.groupby('鄉鎮市區').count().plot(kind='bar',y='編號',figsize=(10,6),fontsize=14,title='各鄉鎮資料筆數')
ax.set_ylabel('資料筆數')
plt.show()


df2 = df[['鄉鎮市區','土地移轉總面積平方公尺','建物移轉總面積平方公尺','單價每平方公尺','建物現況格局-房','建物現況格局-廳','建物現況格局-衛','車位移轉總面積平方公尺','車位總價元','屋齡','建築完成年','交易年','總價元']]
print(df2.groupby('鄉鎮市區').mean())
ax2 = df2.groupby('鄉鎮市區').mean().plot(kind='bar',y='單價每平方公尺',figsize=(10,6),fontsize=14,title='各鄉鎮市單價每平方公尺平均')
plt.show()


# 針對鄉鎮市區做one-hot encoding
df_region = pd.get_dummies(df['鄉鎮市區'])
#df_region.head()


df_ml = pd.merge(df2,df_region,left_index=True,right_index=True)
#df_ml.head()

#以下為去除遺失值與極端值 - 看實際狀況決定要不要做
#df_ml = df_ml.dropna().reset_index(drop=True)
#df_ml = df_ml[df_ml['建物移轉總面積平方公尺']<1000]
#df_ml = df_ml[df_ml['土地移轉總面積平方公尺']<70]
df_ml.plot(kind='scatter',x='建物移轉總面積平方公尺',y='總價元')
plt.show()


'''
預定使用簡單線性迴歸進行資料分析
'''

# 分析前資料處理
# 切分訓練與測試資料
from sklearn.model_selection import train_test_split

X = df_ml[['建物移轉總面積平方公尺']]
y = df_ml[['總價元']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3) # 30% for testing, 70% for training
X_train.head()

# feature 標準化
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_normalized = sc.transform(X_train)
X_test_normalized = sc.transform(X_test)

'''
簡單線性迴歸-使用Scikit-Learn SGDRegressor
'''

from sklearn import datasets, linear_model

#SGDRegressor的input y需要一維矩陣type
y_train_1d = y_train.values.ravel()
y_test_1d = y_test.values.ravel()

# linear regression 物件
sgdr = linear_model.SGDRegressor(max_iter=10,learning_rate='constant',eta0=0.001)

# 訓練模型
sgdr.fit(X_train_normalized, y_train_1d)

print('各變項參數:', sgdr.coef_)
print("MSE: %.2f" % np.mean((sgdr.predict(X_test_normalized) - y_test_1d) ** 2))
print("R Square:",sgdr.score(X_test_normalized,y_test))

plt.scatter(X_train['建物移轉總面積平方公尺'], y_train_1d,  color='blue', marker = 'x')

plt.plot(X_train, sgdr.predict(X_train_normalized), color='green', linewidth=1)

plt.ylabel('總價元(10K)')
plt.xlabel('建物移轉總面積平方公尺')

plt.show()


'''
K折交叉驗證 (K-fold Cross-Validation) + 學習曲線 (Learning Curve)
'''
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure(figsize=(10,6))  #調整作圖大小
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

from sklearn.model_selection import KFold,StratifiedKFold

cv = KFold(n_splits=5, random_state=None, shuffle=True)
estimator = linear_model.SGDRegressor(max_iter=100,learning_rate='constant',eta0=0.01)

sc.fit(X_train)
plot_learning_curve(estimator, "SGDRegressor",
                    sc.transform(X), y.values.ravel(), cv=cv, train_sizes=np.linspace(0.2, 1.0, 5))


# linear regression
regr = linear_model.LinearRegression()

# 訓練模型
regr.fit(X_train_normalized, y_train)

print('各變項參數:', regr.coef_)
print("MSE: %.2f" % np.mean((regr.predict(X_test_normalized) - y_test) ** 2))
print("R Square:",regr.score(X_test_normalized,y_test))

plt.figure(figsize=(10,6))  #調整作圖大小
plt.scatter(X_train['建物移轉總面積平方公尺'], y_train['總價元'],  color='blue', marker = 'x')
plt.plot(X_train['建物移轉總面積平方公尺'], regr.predict(X_train_normalized), color='green', linewidth=1)

plt.ylabel('總價元(10K)')
plt.xlabel('建物移轉總面積平方公尺')

plt.show()


# linear regression
regr = linear_model.LinearRegression()

# 訓練模型
regr.fit(X_train_normalized, y_train)

print('各變項參數:', regr.coef_)
print("MSE: %.2f" % np.mean((regr.predict(X_test_normalized) - y_test) ** 2))
print("R Square:",regr.score(X_test_normalized,y_test))

plt.figure(figsize=(10,6))  #調整作圖大小
plt.scatter(X_train['建物移轉總面積平方公尺'], y_train['總價元'],  color='blue', marker = 'x')
plt.plot(X_train['建物移轉總面積平方公尺'], regr.predict(X_train_normalized), color='green', linewidth=1)

plt.ylabel('總價元(10K)')
plt.xlabel('建物移轉總面積平方公尺')

plt.show()



cv = KFold(n_splits=4, random_state=None, shuffle=True)
estimator = linear_model.LinearRegression()

sc.fit(X_train)
plot_learning_curve(estimator, "LinearRegression", sc.transform(X),
                    y.values.ravel(), cv=cv , train_sizes=np.linspace(0.2, 1.0, 5))


