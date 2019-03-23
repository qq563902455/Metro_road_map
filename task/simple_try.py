import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb

train = pd.read_csv('./processedData/train.csv')
test = pd.read_csv('./processedData/test.csv')
roadmap = pd.read_csv('./rawdata/Metro_roadMap.csv')


train.stationID = pd.Categorical(train.stationID)
test.stationID = pd.Categorical(test.stationID)


train_set_days = [3, 4, 9, 10, 11, 16, 17, 18, 23, 24, 25]
valid_set_days = [8, 15, 22]

train_features = train.drop(['inNums', 'outNums', 'day'], axis=1).columns

train_x = train[train.day.apply(lambda x: x in train_set_days)][train_features]
train_y_in = train[train.day.apply(lambda x: x in train_set_days)]['inNums']
train_y_out = train[train.day.apply(lambda x: x in train_set_days)]['outNums']


valid_x = train[train.day.apply(lambda x: x in valid_set_days)][train_features]
valid_y_in = train[train.day.apply(lambda x: x in valid_set_days)]['inNums']
valid_y_out = train[train.day.apply(lambda x: x in valid_set_days)]['outNums']

test_x = test[train_features]

param_dict = {
    'random_state': 2017,
    'num_leaves': 31,
    'max_depth': 5,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'learning_rate': 0.15,
    'n_estimators': 4000,
    'reg_sqrt': True,
    'objective': 'regression_l1',
}


model_in = lgb.LGBMRegressor(**param_dict)
model_in.fit(train_x, train_y_in, eval_set=[(valid_x, valid_y_in)], early_stopping_rounds=200, verbose=100)

feature_importances_in = pd.Series(model_in.feature_importances_)
feature_importances_in.index = train_x.columns.tolist()
feature_importances_in = feature_importances_in.sort_values()

param_dict = {
    'random_state': 2017,
    'num_leaves': 31,
    'max_depth': 5,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'learning_rate': 0.1,
    'n_estimators': 5000,
    'reg_sqrt': True,
    'objective': 'regression_l1',
}

model_out = lgb.LGBMRegressor(**param_dict)
model_out.fit(train_x, train_y_out, eval_set=[(valid_x, valid_y_out)], early_stopping_rounds=200, verbose=100)

feature_importances_out = pd.Series(model_out.feature_importances_)
feature_importances_out.index = train_x.columns.tolist()
feature_importances_out = feature_importances_out.sort_values()

valid_in_pred = model_in.predict(valid_x)
valid_out_pred = model_out.predict(valid_x)

valid_in_pred[valid_in_pred<0] = 0
valid_out_pred[valid_out_pred<0] = 0

mae_in = (valid_y_in - valid_in_pred).abs().mean()
mae_out = (valid_y_out - valid_out_pred).abs().mean()

print('inNums MAE:\t', mae_in)
print('outNums MAE:\t', mae_out)

test_in_pred = model_in.predict(test_x)
test_out_pred = model_out.predict(test_x)

test_in_pred[test_in_pred<0] = 0
test_out_pred[test_out_pred<0] = 0

test.inNums = test_in_pred
test.outNums = test_out_pred

test[['stationID', 'startTime', 'endTime', 'inNums', 'outNums']].to_csv('./submit.csv', index=False)

# (valid_y_in - valid_x.inNums_lastday).abs().mean()
# (valid_y_in - 0.4*valid_x['inNums_lastday'] - 0.3*valid_x['inNums_lastday_-1']- 0.3*valid_x['inNums_lastday_1']).abs().mean()
#
#
# (valid_y_out - valid_x.outNums_lastday).abs().mean()
# (valid_y_out - 0.4*valid_x['outNums_lastday'] - 0.3*valid_x['outNums_lastday_-1']- 0.3*valid_x['outNums_lastday_1']).abs().mean()
