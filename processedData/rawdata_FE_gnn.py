import pandas as pd
import numpy as np

def feature_extract(train_x, train_y=None):

    train_y_ex = pd.read_csv('./rawdata/Metro_testB/testB_submit_2019-01-27.csv')
    train_y_ex['time'] = pd.to_datetime(train_y_ex.startTime)
    base_time = train_y_ex.time.min().floor('D')
    train_y_ex.time = (train_y_ex.time - base_time).dt.seconds//600

    if train_y is not None:
        train_y_ex = train_y_ex.drop(['startTime', 'endTime'], axis=1)


    train_x.time = pd.to_datetime(train_x.time)
    base_time = train_x.time.min().floor('D')
    train_x.time = (train_x.time - base_time).dt.seconds//600

    x_group_count = train_x[['time', 'stationID', 'status', 'payType', 'userID']].groupby(by=['time', 'stationID', 'status', 'payType']).count()
    x_group_count = x_group_count.rename({'userID': 'num'}, axis=1)
    x_group_count = x_group_count.reset_index()


    if train_y is not None:
        train_y.time = pd.to_datetime(train_y.time)
        base_time = train_y.time.min().floor('D')
        train_y.time = (train_y.time - base_time).dt.seconds//600

        y_group_count = train_y[['time', 'stationID', 'status', 'payType', 'userID']].groupby(by=['time', 'stationID', 'status', 'payType']).count()
        y_group_count = y_group_count.rename({'userID': 'num'}, axis=1)
        y_group_count = y_group_count.reset_index()

        xy_group = pd.merge(left=x_group_count, right=y_group_count, how='outer', on=['time', 'stationID', 'status', 'payType'])

        train_y_ex.outNums = 0
        train_y_ex.inNums = 0
        for payType in [0, 1, 2, 3]:
            train_y_ex = pd.merge(train_y_ex, xy_group[np.array(xy_group.status==0)&np.array(xy_group.payType==payType)], how='left', on=['stationID', 'time'])
            train_y_ex = train_y_ex.fillna(0)
            train_y_ex.outNums += train_y_ex.num_y
            train_y_ex = train_y_ex.drop(['status', 'payType', 'num_y'], axis=1)

            train_y_ex = pd.merge(train_y_ex, xy_group[np.array(xy_group.status==1)&np.array(xy_group.payType==payType)], how='left', on=['stationID', 'time'])
            train_y_ex = train_y_ex.fillna(0)
            train_y_ex.inNums += train_y_ex.num_y
            train_y_ex = train_y_ex.drop(['status', 'payType', 'num_y'], axis=1)

            train_y_ex =  train_y_ex.rename({'num_x_x': 'outNums_lastday_'+str(payType), 'num_x_y': 'inNums_lastday_'+str(payType)}, axis=1)
    else:
        xy_group = x_group_count
        for payType in [0, 1, 2, 3]:
            train_y_ex = pd.merge(train_y_ex, xy_group[np.array(xy_group.status==0)&np.array(xy_group.payType==payType)], how='left', on=['stationID', 'time'])
            train_y_ex = train_y_ex.drop(['status', 'payType'], axis=1)

            train_y_ex = pd.merge(train_y_ex, xy_group[np.array(xy_group.status==1)&np.array(xy_group.payType==payType)], how='left', on=['stationID', 'time'])
            train_y_ex = train_y_ex.drop(['status', 'payType'], axis=1)

            train_y_ex =  train_y_ex.rename({'num_x': 'outNums_lastday_'+str(payType), 'num_y': 'inNums_lastday_'+str(payType)}, axis=1)


    train_y_ex = train_y_ex.fillna(0)
    return train_y_ex


def multidays_feature_extract(day1, day2):
    feature_names = []
    for col in day1.columns:
        if 'lastday' in col:
            feature_names.append(col)

    out = pd.merge(day1, day2[['stationID', 'time']+feature_names], on=['stationID', 'time'], how='left')
    return out


roadmap = pd.read_csv('./rawdata/Metro_roadMap.csv')


train_list = []
for i in range(1, 26):

    x_day_1 = str(i)
    if len(x_day_1) == 1: x_day_1 = '0' + x_day_1

    x_day_2 = str(i-6)
    if len(x_day_2) == 1: x_day_2 = '0' + x_day_2

    y_day = str(i+1)
    if len(y_day) == 1: y_day = '0' + y_day

    if i != 25:
        train_x_1 = pd.read_csv('./rawdata/Metro_train/record_2019-01-'+x_day_1+'.csv')
        # train_x_2 = pd.read_csv('./rawdata/Metro_train/record_2019-01-'+x_day_2+'.csv')

        train_y = pd.read_csv('./rawdata/Metro_train/record_2019-01-'+y_day+'.csv')
    else:
        train_x_1 = pd.read_csv('./rawdata/Metro_train/record_2019-01-'+x_day_1+'.csv')
        # train_x_2 = pd.read_csv('./rawdata/Metro_train/record_2019-01-'+x_day_2+'.csv')

        train_y = pd.read_csv('./rawdata/Metro_testB/testB_record_2019-01-26.csv')

    # train_1 = feature_extract(train_x_1, train_y)
    # train_2 = feature_extract(train_x_2, train_y)
    #
    # train = multidays_feature_extract(train_1, train_2)

    train = feature_extract(train_x_1, train_y)

    train['day'] = i+1

    train_list.append(train)
    print(i)

train = pd.concat(train_list)

test_x_1 = pd.read_csv('./rawdata/Metro_testB/testB_record_2019-01-26.csv')
# test_x_2 = pd.read_csv('./rawdata/Metro_train/record_2019-01-22.csv')

# test_1 = feature_extract(test_x_1)
# test_2 = feature_extract(test_x_2)

# test = multidays_feature_extract(test_1, test_2)

test = feature_extract(test_x_1)


test['day'] = 27

train.to_csv('./processedData/train.csv', index=False)
test.to_csv('./processedData/test.csv', index=False)
