import pandas as pd
import numpy as np

def feature_extract(train_x, train_y=None):

    train_y_ex = pd.read_csv('./rawdata/Metro_testA/testA_submit_2019-01-29.csv')
    train_y_ex['time'] = pd.to_datetime(train_y_ex.startTime)
    base_time = train_y_ex.time.min().floor('D')
    train_y_ex.time = (train_y_ex.time - base_time).dt.seconds//600

    if train_y is not None:
        train_y_ex = train_y_ex.drop(['startTime', 'endTime'], axis=1)


    train_x.time = pd.to_datetime(train_x.time)
    base_time = train_x.time.min().floor('D')
    train_x.time = (train_x.time - base_time).dt.seconds//600

    x_group_count = train_x[['time', 'stationID', 'status', 'payType']].groupby(by=['time', 'stationID', 'status']).count()
    x_group_count = x_group_count.rename({'payType': 'num'}, axis=1)
    x_group_count = x_group_count.reset_index()


    if train_y is not None:
        train_y.time = pd.to_datetime(train_y.time)
        base_time = train_y.time.min().floor('D')
        train_y.time = (train_y.time - base_time).dt.seconds//600

        y_group_count = train_y[['time', 'stationID', 'status', 'payType']].groupby(by=['time', 'stationID', 'status']).count()
        y_group_count = y_group_count.rename({'payType': 'num'}, axis=1)
        y_group_count = y_group_count.reset_index()

        xy_group = pd.merge(left=x_group_count, right=y_group_count, how='outer', on=['time', 'stationID', 'status'])


        train_y_ex = pd.merge(train_y_ex, xy_group[xy_group.status==0], how='left', on=['stationID', 'time'])
        train_y_ex.outNums = train_y_ex.num_y
        train_y_ex = train_y_ex.drop(['status', 'num_y'], axis=1)

        train_y_ex = pd.merge(train_y_ex, xy_group[xy_group.status==1], how='left', on=['stationID', 'time'])
        train_y_ex.inNums = train_y_ex.num_y
        train_y_ex = train_y_ex.drop(['status', 'num_y'], axis=1)

        train_y_ex =  train_y_ex.rename({'num_x_x': 'outNums_lastday', 'num_x_y': 'inNums_lastday'}, axis=1)
    else:
        xy_group = x_group_count
        train_y_ex = pd.merge(train_y_ex, xy_group[xy_group.status==0], how='left', on=['stationID', 'time'])
        train_y_ex = train_y_ex.drop(['status'], axis=1)

        train_y_ex = pd.merge(train_y_ex, xy_group[xy_group.status==1], how='left', on=['stationID', 'time'])
        train_y_ex = train_y_ex.drop(['status'], axis=1)

        train_y_ex =  train_y_ex.rename({'num_x': 'outNums_lastday', 'num_y': 'inNums_lastday'}, axis=1)


    train_y_ex['day'] = i+1
    train_y_ex = train_y_ex.fillna(0)

    out_df = train_y_ex

    for delay in [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]:

        train_temp = train_y_ex.copy()
        train_temp.time = train_temp.time + delay
        train_temp = train_temp[train_temp.time>=0]
        train_temp = train_temp[train_temp.time<=80]

        train_temp = train_temp[['outNums_lastday', 'inNums_lastday', 'stationID', 'time', 'day']]

        train_temp = train_temp.rename({'outNums_lastday':'outNums_lastday_'+str(delay), 'inNums_lastday': 'inNums_lastday_'+ str(delay)}, axis=1)

        out_df = pd.merge(out_df, train_temp, how='left', on=['stationID', 'time', 'day'])

        out_df.loc[out_df['outNums_lastday_'+str(delay)].isna(), 'outNums_lastday_'+str(delay)] = out_df[out_df['outNums_lastday_'+str(delay)].isna()]['outNums_lastday']
        out_df.loc[out_df['inNums_lastday_'+str(delay)].isna(), 'inNums_lastday_'+ str(delay)] = out_df[out_df['inNums_lastday_'+ str(delay)].isna()]['inNums_lastday']

        # out_df['outNums_lastday_'+str(delay)] = out_df['outNums_lastday_'+str(delay)] - out_df['outNums_lastday']
        # out_df['inNums_lastday_'+str(delay)] = out_df['inNums_lastday_'+str(delay)] - out_df['inNums_lastday']

    neighbor_list = []
    for stationID in out_df.stationID.unique():

        neighbor_stID = roadmap.index[roadmap[str(stationID)]==1].tolist()
        neighbor_train = out_df[out_df.stationID.apply(lambda x : x in neighbor_stID)]
        features_keep = ['time']
        for col in neighbor_train:
            if 'lastday' in col:
                features_keep.append(col)
        neighbor_train = neighbor_train[features_keep]

        neighbor_train = neighbor_train.groupby(by='time').mean()
        neighbor_train = neighbor_train.reset_index()

        feature_new_names = []
        for col in neighbor_train.columns:
            if 'lastday' in col:
                col = col.replace('lastday', 'neighbor')
            feature_new_names.append(col)

        neighbor_train.columns = feature_new_names
        neighbor_train['stationID'] = stationID
        neighbor_list.append(neighbor_train)

    neighbor_df = pd.concat(neighbor_list)

    out_df = pd.merge(out_df, neighbor_df, on=['time', 'stationID'])

    return out_df

roadmap = pd.read_csv('./rawdata/Metro_roadMap.csv')


train_list = []
for i in range(1, 25):

    x_day = str(i)
    if len(x_day) == 1: x_day = '0' + x_day
    y_day = str(i+1)
    if len(y_day) == 1: y_day = '0' + y_day

    train_x = pd.read_csv('./rawdata/Metro_train/record_2019-01-'+x_day+'.csv')
    train_y = pd.read_csv('./rawdata/Metro_train/record_2019-01-'+y_day+'.csv')

    train = feature_extract(train_x, train_y)

    train_list.append(train)
    print(i)

train = pd.concat(train_list)

test_x = pd.read_csv('./rawdata/Metro_testA/testA_record_2019-01-28.csv')
test = feature_extract(test_x)

train.to_csv('./processedData/train.csv', index=False)
test.to_csv('./processedData/test.csv', index=False)
