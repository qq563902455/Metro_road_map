import pandas as pd
import numpy as np

submit_1254 = pd.read_csv('./modelFusion/submit_1254.csv')
submit_1280 = pd.read_csv('./modelFusion/submit_1280.csv')


submit_final = submit_1254.copy()

submit_final.inNums = 0.7*submit_1254.inNums + 0.3*submit_1280.inNums
submit_final.outNums = 0.7*submit_1254.outNums + 0.3*submit_1280.outNums


submit_final[['stationID', 'startTime', 'endTime', 'inNums', 'outNums']].to_csv('./modelFusion/submit_final.csv', index=False)
