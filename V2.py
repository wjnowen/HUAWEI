#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import lightgbm as lgb


# In[4]:


pd.set_option('display.max_columns',None)


# In[92]:


#读取数据
age_train = pd.read_csv("age_train.csv", names=['uid','age_group'])
age_test = pd.read_csv("age_test.csv", names=['uid'])
user_basic_info = pd.read_csv("user_basic_info.csv", names=['uid','gender','city','prodName','ramCapacity','ramLeftRation','romCapacity','romLeftRation','color','fontSize','ct','carrier','os'])
user_behavior_info = pd.read_csv("user_behavior_info.csv", names=['uid','bootTimes','AFuncTimes','BFuncTimes','CFuncTimes','DFuncTimes','EFuncTimes','FFuncTimes','FFuncSum'])
user_app_actived = pd.read_csv("user_app_actived.csv", names=['uid','appId'])
app_info = pd.read_csv("app_info.csv", names=['appId', 'category'])




#获取种类数
def f(x):
    s = x.value_counts()
    return np.nan if len(s) == 0 else s.index[0]
    
user_app_num = {}
app_cate_dict = {}
user_app_num_sort = {}    

def processUserAppUsage():
    reader = pd.read_csv('./user_app_usage.csv',iterator=True,names=['uid','appId','duration','times','use_date'])
    chunkSize = 10000000
    loop = True
    allchunk = pd.DataFrame()
    
    app_info = pd.read_csv("app_info.csv", names=['appId','category'])
    cats = list(set(app_info['category']))
    category2id = dict(zip(sorted(cats), range(0,len(cats))))
    id2category = dict(zip(range(0,len(cats)), sorted(cats)))
    app_info['category'] = app_info['category'].apply(lambda x: category2id[x])
    
    while loop:
        try:
            chunk = reader.get_chunk(chunkSize)
            chunk = optimize_memory(chunk)
            #得到平均每次使用时间
            chunk['duration_average'] = chunk['duration']/chunk['times']
            print(len(chunk),mem_usage(chunk))
            chunk = chunk.merge(app_info, how='left', on='appId')
            chunk = chunk.merge(chunk.groupby('uid')['category'].nunique().to_frame(), how='left', on='uid')
            chunk['usage_most_used_category'] = chunk.groupby(['uid'])['category'].transform(f)
            #读取用户基本信息
            chunk = chunk.merge(user_basic_info,on=['uid'],how='left')
            chunk.drop(columns=['uid'],axis=1,inplace=True)
            print(len(chunk),mem_usage(chunk))
            for col in ['duration','duration_average','times','gender','ramCapacity',
                                                               'ramLeftRation','romCapacity','romLeftRation',
                                                              'fontSize','os']:
                _ = chunk.groupby(['appId'],as_index=False)[col].agg({col+'_max':'max',col+'_min':'min',col+'_mean':'mean'})
                chunk = chunk.merge(_,on=['appId'],how='left')
                chunk.drop(columns=[col],axis=1,inplace=True)
            

            chunk = chunk.drop_duplicates(['appId'])
            allchunk = pd.concat([allchunk,chunk],ignore_index=True)
            print("allchunk:",len(allchunk),mem_usage(allchunk))
        except StopIteration:
            loop = False

    #去掉使用日期特征
    allchunk.drop(columns=['use_date'],axis=1,inplace=True)    
    #debug
    allchunk.columns

    #去掉appid
    allchunk = allchunk.drop_duplicates(['appId'])
    #再次求特征
    allchunk['number_sort'] = allchunk['appId'].apply(lambda x:user_app_num_sort[x] if x in user_app_num_sort else np.nan)
    allchunk['app_cate'] = allchunk['appId'].apply(lambda x:app_cate_dict[x][0] if x in app_cate_dict else 'unknow')
    #debug
    allchunk.head()
    #保存
    allchunk.to_csv('./new.csv',index=False)

	#去掉四种特征数据
    user_basic_info.drop(columns=['prodName','color','gender','carrier'],inplace=True)
    




processUserAppUsage()



# In[8]:


#处理app使用相关数据
#对user_app_actived.csv简单统计
#将之前训练的appuseProcess.csv进行合并
def mergeAppData(baseTable):
    resTable = baseTable.merge(user_app_actived, how='left', on='uid')
    resTable['appId'] = resTable['appId'].apply(lambda x: len(list(x.split('#'))))
    appusedTable = pd.read_csv("./new.csv")
    resTable = resTable.merge(appusedTable, how='left', on='uid')
    resTable[['category', 'usage_most_used_category']] = resTable[['category', 'usage_most_used_category']].fillna(41)
    resTable = resTable.fillna(0)

    resTable['os'] = resTable['os'].apply(lambda x: x if type(x)==str else str(x))
    sort_temp = sorted(list(set(resTable['os'])))  
    class2id['os2id'] = dict(zip(sort_temp, range(1, len(sort_temp)+1)))
    resTable['os'] = resTable['os'].apply(lambda x: class2id['os2id'][x])
    return resTable


# In[9]:


#合并用户基本特征以及app使用相关特征，作为训练集和测试集
df_train = mergeAppData(age_train)
df_test = mergeAppData(age_test)


# In[11]:


print(df_train)
print(df_test)


# In[142]:


df_train.shape


# In[15]:


#训练模型

from sklearn.feature_selection import SelectFromModel, VarianceThreshold, SelectKBest, chi2, mutual_info_classif, f_classif
from sklearn.preprocessing import Imputer
from sklearn.ensemble import ExtraTreesClassifier

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold,KFold


# In[16]:


X = df_train.drop(['age_group','uid'], axis=1)
y = df_train['age_group']
uid = df_test['uid']
test = df_test.drop('uid', axis=1)

skf = StratifiedKFold(n_splits=4, random_state=1030, shuffle=True)
for train, test in skf.split(X,y):
    print('Train: %s | test: %s' % (train, test))


# In[25]:


print("训练模型：")
param = {
        'learning_rate': 0.14,
        'lambda_l1': 1,
        'lambda_l2': 0.01,
        'max_depth': 15,
        'objective': 'multiclass',
        'num_class': 7,
        'num_leaves': 31,
        'min_data_in_leaf': 50,
        'max_bin': 230,
        'feature_fraction': 0.8,
        'metric': 'multi_error'
        }

X = df_train.drop(['age_group','uid'], axis=1)
y = df_train['age_group']
uid = df_test['uid']
test = df_test.drop('uid', axis=1)

xx_score = []
cv_pred = []
skf = StratifiedKFold(n_splits=4, random_state=1030, shuffle=True)
for index, (train_index, vali_index) in enumerate(skf.split(X, y)):
    print(index)
    x_train, y_train, x_vali, y_vali = np.array(X)[train_index], np.array(y)[train_index], np.array(X)[vali_index], np.array(y)[vali_index]
    train = lgb.Dataset(x_train, y_train)
    vali =lgb.Dataset(x_vali, y_vali)
    print("开始训练...")
    model = lgb.train(param, train, num_boost_round=10, valid_sets=[vali], early_stopping_rounds=50)
    print("开始预测...")
    xx_pred = model.predict(x_vali,num_iteration=model.best_iteration)
    print (xx_pred)
    print (xx_pred.shape)
    xx_pred = [np.argmax(x) for x in xx_pred]
    print (xx_pred)
    print("预测训练集的x:")
    xx_score.append(f1_score(y_vali,xx_pred,average='weighted'))
    y_test = model.predict(test,num_iteration=model.best_iteration)
    y_test = [np.argmax(x) for x in y_test]
    if index == 0:
        cv_pred = np.array(y_test).reshape(-1, 1)
    else:
        cv_pred = np.hstack((cv_pred, np.array(y_test).reshape(-1, 1)))
        
submit = []
for line in cv_pred:
    submit.append(np.argmax(np.bincount(line)))
df = pd.DataFrame({'id':uid.as_matrix(),'label':submit})
df.to_csv('submission.csv',index=False)


# In[19]:


age_train['age_group'].nunique()


# In[28]:


print(xx_score)




