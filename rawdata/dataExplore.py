import pandas as pd
import numpy as np

data0128 = pd.read_csv('./rawdata/Metro_testA/testA_record_2019-01-28.csv')
data0128.time = pd.to_datetime(data0128.time)-pd.to_datetime('2019-01-28 00:00:00')
data0128.time = data0128.time.dt.seconds//600

temp = data0128[['time', 'stationID', 'status', 'payType']].groupby(by=['time', 'stationID', 'status']).count()
temp = temp.rename({'payType': 'num'}, axis=1)
temp = temp.reset_index()



submit = pd.read_csv('./rawdata/Metro_testA/testA_submit_2019-01-29.csv')
submit['time'] = pd.to_datetime(submit.startTime) - pd.to_datetime('2019-01-29 00:00:00')
submit.time = submit.time.dt.seconds//600


submit = pd.merge(left=submit, right=temp[temp.status==0], on=['stationID', 'time'], how='left')
submit.outNums = submit.num
submit = submit.drop(['status', 'num'], axis=1)

submit = pd.merge(left=submit, right=temp[temp.status==1], on=['stationID', 'time'], how='left')
submit.inNums = submit.num
submit = submit.drop(['status', 'num', 'time'], axis=1)

submit = submit.fillna(0)

submit.to_csv('./submit.csv', index=False)
