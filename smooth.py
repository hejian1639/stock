from __future__ import print_function

import datetime

import matplotlib.pyplot as plt
import pandas as pd
from dateutil.relativedelta import relativedelta

df = pd.read_csv('portland-oregon-average-monthly-.csv', index_col=0)
df.index.name = None  # 将index的name取消
df.reset_index(inplace=True)

start = datetime.datetime.strptime("1960-01", "%Y-%m")  # 把一个时间字符串解析为时间元组
date_list = [start + relativedelta(months=x) for x in range(0, 114)]  # 从1973-01-01开始逐月增加组成list
df['index'] = date_list
df.set_index(['index'], inplace=True)
df.index.name = None
df.columns = ['riders']
df.riders.plot(figsize=(12, 8), title='Monthly Ridership', fontsize=14)
plt.show()

df = pd.read_csv('portland-oregon-average-monthly-.csv', index_col=0)
df.index.name = None  # 将index的name取消
df.reset_index(inplace=True)
smooth_array = df.iloc[:, 1]

k = 10


def average(array, index):
    sum = 0
    begin = index - k
    end = index + k
    if begin < 0:
        begin = 0

    if end > len(array):
        end = len(array)

    for i in range(begin, end):
        sum += array[i]

    sum /= end - begin
    return sum


for i in range(0, len(smooth_array)):
    smooth_array[i] = average(smooth_array, i)

start = datetime.datetime.strptime("1960-01", "%Y-%m")  # 把一个时间字符串解析为时间元组
date_list = [start + relativedelta(months=x) for x in range(0, 114)]  # 从1973-01-01开始逐月增加组成list
print(df.size)
df['index'] = date_list
df.set_index(['index'], inplace=True)
df.index.name = None
df.columns = ['riders']
df.riders.plot(figsize=(12, 8), title='Monthly Ridership', fontsize=14)
plt.show()

df = pd.read_csv('portland-oregon-average-monthly-.csv', index_col=0)
df.index.name = None  # 将index的name取消
df.reset_index(inplace=True)
array = df.iloc[:, 1]

for i in range(0, len(array)):
    array[i] -= smooth_array[i]

print(array)

start = datetime.datetime.strptime("1960-01", "%Y-%m")  # 把一个时间字符串解析为时间元组
date_list = [start + relativedelta(months=x) for x in range(0, 114)]  # 从1973-01-01开始逐月增加组成list
print(df.size)
df['index'] = date_list
df.set_index(['index'], inplace=True)
df.index.name = None
df.columns = ['riders']
df.riders.plot(figsize=(12, 8), title='Monthly Ridership', fontsize=14)
plt.show()
