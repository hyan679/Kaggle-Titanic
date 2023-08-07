import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

def ticket_prefix(x):
    if x.count(' '):
        return x.split()[0].replace('.', '').replace('/', '')

def fit_transform(train_x, test_x):
    train_num = train_x.shape[0]
    data = pd.concat([train_x, test_x], axis='index')

    # Cabin
    data['Cabin'] = data['Cabin'].map(arg=lambda x:x[0], na_action='ignore')
    # Ticket
    data['Ticket'] = data['Ticket'].map(arg=ticket_prefix, na_action='ignore')
    # relatives
    data['Relatives'] = data['SibSp'] + data['Parch']

    data.drop(labels=['PassengerId','Name'], axis='columns', inplace=True)
    data = pd.get_dummies(data)
    data.fillna(value=0, inplace=True)
    tran_pp, test_pp = data.iloc[:train_num,:], data.iloc[train_num:,:]
    # Age
    age_mean = tran_pp['Age'].mean()
    tran_pp['Age'].fillna(value=age_mean, inplace=True)
    test_pp['Age'].fillna(value=age_mean, inplace=True)
    return tran_pp, test_pp
    #return data.loc[:train_num], data.loc[train_num:]
    
train_x = pd.read_csv('data/train.csv')
test_x = pd.read_csv('data/test.csv')

train_y = train_x.pop('Survived')
res_id = test_x['PassengerId']


train_num = train_x.shape[0]
data = pd.concat([train_x, test_x], axis='index')
# Cabin
data['Cabin'] = data['Cabin'].map(arg=lambda x:x[0], na_action='ignore')
# Ticket
data['Ticket'] = data['Ticket'].map(arg=ticket_prefix, na_action='ignore')
# relatives
data['Relatives'] = data['SibSp'] + data['Parch']
data.drop(labels=['PassengerId','Name'], axis='columns', inplace=True)
data = pd.get_dummies(data)
data.fillna(value=0, inplace=True)
train_pp, test_pp = data.iloc[:train_num,:].copy(), data.iloc[train_num:,:].copy()
# Age
age_mean = train_pp['Age'].mean()
train_pp['Age'].fillna(value=age_mean, inplace=True)
test_pp['Age'].fillna(value=age_mean, inplace=True)

gbc = GradientBoostingClassifier(n_estimators=90, learning_rate=0.25)
gbc.fit(train_pp, train_y)
#Predition
res = gbc.predict(test_pp)
res = pd.Series(res, name='Survived')
res = pd.concat([res_id, res], axis='columns')
res.to_csv('res.csv', index=False)