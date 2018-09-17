# import lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



#define missing data
def missing_data(dataset):
    total = dataset.isnull().sum().sort_values(ascending=False)
    percent_1 = dataset.isnull().sum()/dataset.isnull().count()*100
    percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
    return pd.concat([total, percent_2], axis=1, keys=['筆數', '%'])


# 相關性分析將dataframe的背景色highlight
def highlight_highcorr(s):
    is_high = ((s >= 0.6) & (s < 1))
    return ['background-color: yellow' if v else '' for v in is_high]



# define style
plt.rcParams['font.family']='SimHei' #顯示中文
plt.rcParams['axes.unicode_minus']=False #正常顯示負號
pd.set_option("display.max_columns",50) #設定pandas最多顯示出50個欄位資訊

# create dataset
df = pd.read_csv('data/A_LVR_LAND_A.csv' , encoding='big5')


df['都市土地使用分區'] = df['都市土地使用分區'].replace({'住':1,'其他':2,'商':3,
                                '工':4,'農':5})

df.loc[:,'總價元'] = df.loc[:,'總價元']/10000 #改成以萬為單位，方便圖表顯示

print(df.loc[0,'交易年月日'])
print(type(df.loc[0,'交易年月日']))

for i in range(len(df)):
    df.loc[i,'交易年'] = round(df.loc[i,'交易年月日']/10000)


#print(i)

from datetime import datetime
now = datetime.now()
now.year #(年,月,日,時,分,秒,微秒) ps: 微秒(microseconds)=1/1000000秒

for i in range(len(df)):
    df.loc[i,'建築完成年'] = round(df.loc[i,'建築完成年月']/10000)
    df.loc[i,'屋齡'] = now.year - 1911 - round(df.loc[i,'建築完成年月']/10000)


corr = df[['總價元','鄉鎮市區','建物型態','都市土地使用分區','土地移轉總面積平方公尺','建物移轉總面積平方公尺',
           '建物現況格局-房','建物現況格局-廳','建物現況格局-衛','車位移轉總面積平方公尺','車位總價元','屋齡','建築完成年','交易年']].corr()


plt.figure(figsize=(10,10))
sns.heatmap(corr, square=True, annot=True)
plt.show()


df.plot(kind='scatter',title='散佈圖',figsize=(10,6),x='建物現況格局-房',y='總價元',marker='+')
plt.show()