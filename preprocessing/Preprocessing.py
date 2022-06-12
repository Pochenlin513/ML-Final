import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.decomposition import NMF, PCA

NUM_FEATS = ['職等','管理層級','專案時數','專案總數','特殊專案佔比',
             '訓練時數A','訓練時數B','訓練時數C','生產總額','榮譽數',
             '升遷速度','近三月請假數A','近一年請假數A','近三月請假數B',
             '近一年請假數B','出差數A','出差數B','出差集中度','年度績效等級A',
             '年度績效等級B','年度績效等級C','年齡層級','年資層級A',
             '年資層級B','年資層級C','任職前工作平均年數','眷屬量','通勤成本',
            '加班數 Q1','出差數A Q1','出差數B Q1','請假數A Q1','請假數B Q1', '出差數A+B Q1', '請假數A+B Q1',
             '加班數 Q2','出差數A Q2','出差數B Q2','請假數A Q2','請假數B Q2', '出差數A+B Q2', '請假數A+B Q2',
             '加班數 Q3','出差數A Q3','出差數B Q3','請假數A Q3','請假數B Q3', '出差數A+B Q3', '請假數A+B Q3',
             '加班數 Q4','出差數A Q4','出差數B Q4','請假數A Q4','請假數B Q4', '出差數A+B Q4', '請假數A+B Q4',
             '訓練時數A+B+C', '近三月請假數A+B', '近一年請假數A+B', '出差數A+B', '年度績效等級A+B+C', '年資層級A+B+C'
            ]
CAT_FEATS = ['sex','工作分類','廠區代碼','工作資歷1','工作資歷2',
             '工作資歷3','工作資歷4','工作資歷5','當前專案角色',
             '工作地點','是否升遷','婚姻狀況','畢業科系類別','歸屬部門',]

def RemoveData(ds, ds_test, cols=['最高學歷', '畢業學校類別']):
    columns = list(ds.columns)
    for col in cols:
        columns.remove(col)
    ds = ds.loc[:,columns]
    ds_test = ds_test.loc[:,columns]
    
    r, _ = np.where(ds.isna())
    r = list(set(r))
    for i in r:
        ds = ds.drop([i])
    return ds, ds_test

def seasonAdd(ds_season):
    AB = ds_season['出差數A'] + ds_season['出差數B']
    AB.name = '出差數A+B'
    ds_season = pd.concat([ds_season, AB], axis=1)           # add to original feat set
    
    AB = ds_season['請假數A'] + ds_season['請假數B']
    AB.name = '請假數A+B'
    ds_season = pd.concat([ds_season, AB], axis=1)           # add to original feat set
    return ds_season

def subData(ds_season, year, Q):
    sub = ds_season.loc[(ds_season['periodQ'] == year + Q)]
    sub.rename(columns={'加班數': '加班數 ' + Q, 
                           '出差數A': '出差數A ' + Q,
                           '出差數B': '出差數B ' + Q,
                           '請假數A': '請假數A ' + Q,
                           '請假數B': '請假數B ' + Q,
                           '出差數A+B': '出差數A+B ' + Q,
                           '請假數A+B': '請假數A+B ' + Q}, inplace=True)
    columns = list(sub.columns)
    columns.remove('periodQ')
    return sub.loc[:,columns]

def subDataYear(ds_season, year):
    sub = subData(ds_season, year, 'Q1')
    for q in ['Q2', 'Q3', 'Q4']:
        sub_new = subData(ds_season, year, q)
        columns = list(sub_new.columns)
        columns.remove('yyyy')
        sub =  pd.merge(sub,sub_new.loc[:,columns], on='PerNo')
    return sub

def mergeSeason(ds_train, ds_test, ds_season):
    ds_season = seasonAdd(ds_season)                         # create A+B...
    
    # for training
    sub = subDataYear(ds_season, '2014')
    for i in ['2015', '2016', '2017']:
        sub_new = subDataYear(ds_season, i)
        sub = pd.concat([sub,sub_new],axis=0,ignore_index=True)
    train_merge = pd.merge(ds_train, sub,  how='left', left_on=['yyyy','PerNo'], right_on = ['yyyy','PerNo'])
    
    sub = subDataYear(ds_season, '2018')                     # 2018 for testing
    test_merge = pd.merge(ds_test, sub,  how='left', left_on=['yyyy','PerNo'], right_on = ['yyyy','PerNo'])
    
    return train_merge, test_merge

def TrainAdd(ds_train):                                      # add up (A, B, C)s in train
    # 訓練時數A,訓練時數B,訓練時數C
    AB = ds_train['訓練時數A'] + ds_train['訓練時數B'] + ds_train['訓練時數C']
    AB.name = '訓練時數A+B+C'
    ds_train = pd.concat([ds_train, AB], axis=1)             # add to original feat set
    
    AB = ds_train['近三月請假數A'] + ds_train['近三月請假數B']
    AB.name = '近三月請假數A+B'
    ds_train = pd.concat([ds_train, AB], axis=1)             # add to original feat set
    
    AB = ds_train['近一年請假數A'] + ds_train['近一年請假數B']
    AB.name = '近一年請假數A+B'
    ds_train = pd.concat([ds_train, AB], axis=1)
    
    AB = ds_train['出差數A'] + ds_train['出差數B']
    AB.name = '出差數A+B'
    ds_train = pd.concat([ds_train, AB], axis=1)
    
    AB = ds_train['年度績效等級A'] + ds_train['年度績效等級B'] + ds_train['年度績效等級C']
    AB.name = '年度績效等級A+B+C'
    ds_train = pd.concat([ds_train, AB], axis=1)
    
    AB = ds_train['年資層級A'] + ds_train['年資層級B'] + ds_train['年資層級C']
    AB.name = '年資層級A+B+C'
    ds_train = pd.concat([ds_train, AB], axis=1)
    
    return ds_train

def stdNormalize(df_train, df_test, NUM_FEATS):
    for feat in NUM_FEATS:
        mean = df_train[feat].mean()
        std = df_train[feat].std()
        df_train[feat] = (df_train[feat] - mean) / std
        df_test[feat] = (df_test[feat] - mean) / std
    return df_train, df_test

def maxminNormalize(df_train, df_test, NUM_FEATS):
    for feat in NUM_FEATS:
        mini = df_train[feat].min()
        maxi = df_train[feat].max()
        df_train[feat] = (df_train[feat] - mini) / (maxi - mini)
        df_test[feat] = (df_test[feat] - mini) / (maxi - mini)
    return df_train, df_test


# input: raw data of type pandas.core.frame.DataFrame
# output: feat & labels
def splitXY(ds):
    X = ds.loc[:,ds.columns[3:]]
    Y = ds.PerStatus
    return X, Y

def keepMutual(x_train, y_train, x_test, k=10):              # keep top-k feat
    selection = SelectKBest(mutual_info_classif, k=k).fit(x_train, y_train)
    features = x_train.columns[selection.get_support()]
    return x_train[features], x_test[features], features

def keepChi(x_train, y_train, x_test, k=10):                 # keep top-k feat
    selection = SelectKBest(chi2, k=k).fit(x_train, y_train)
    features = x_train.columns[selection.get_support()]
    return x_train[features], x_test[features], features

def keepANOVA(x_train, y_train, x_test, k=10):               # keep top-k feat
    selection = SelectKBest(f_classif, k=k).fit(x_train, y_train)
    features = x_train.columns[selection.get_support()]
    return x_train[features], x_test[features], features

def keepNMF(x_train, x_test, k):                             # keep k-dim feat
    nmf = NMF(n_components=k)
    trans_train = nmf.fit_transform(x_train)
    trans_test = nmf.transform(x_test)
    return pd.DataFrame(trans_train), pd.DataFrame(trans_test)

def keepPCA(x_train, x_test, k):                             # keep k-dim feat
    pca = PCA(n_components=k)
    trans_train = pca.fit_transform(x_train)
    trans_test = pca.transform(x_test)    
    return pd.DataFrame(trans_train), pd.DataFrame(trans_test)