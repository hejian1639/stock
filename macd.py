#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tushare
import pandas as pd
import time


def getstartdate(code):
    """obtain establish date of stock"""
    df = tushare.get_stock_basics()
    date = df.ix[code]['timeToMarket']  # 上市日期YYYYMMDD
    return date


def get_EMA(df, N):
    for i in range(len(df)):
        if i == 0:
            df.ix[i, 'ema'] = df.ix[i, 'close']
        if i > 0:
            df.ix[i, 'ema'] = (2 * df.ix[i, 'close'] + (N - 1) * df.ix[i - 1, 'ema']) / (N + 1)
    ema = list(df['ema'])
    return ema


def get_MACD(df, short=12, long=26, M=9):
    a = get_EMA(df, short)
    b = get_EMA(df, long)
    df['diff'] = pd.Series(a) - pd.Series(b)
    # print(df['diff'])
    for i in range(len(df)):
        if i == 0:
            df.ix[i, 'dea'] = df.ix[i, 'diff']
        if i > 0:
            df.ix[i, 'dea'] = (2 * df.ix[i, 'diff'] + (M - 1) * df.ix[i - 1, 'dea']) / (M + 1)
    df['macd'] = 2 * (df['diff'] - df['dea'])
    return df


def cal_macd_system(data, short_, long_, m):
    '''
    data是包含高开低收成交量的标准dataframe
    short_,long_,m分别是macd的三个参数
    返回值是包含原始数据和diff,dea,macd三个列的dataframe
    '''
    data['diff'] = data['close'].ewm(adjust=False, alpha=2 / (short_ + 1), ignore_na=True).mean() - \
                   data['close'].ewm(adjust=False, alpha=2 / (long_ + 1), ignore_na=True).mean()
    data['dea'] = data['diff'].ewm(adjust=False, alpha=2 / (m + 1), ignore_na=True).mean()
    data['macd'] = 2 * (data['diff'] - data['dea'])
    return data


if __name__ == '__main__':
    """arguments: 
    code
    start date
    end date
    index
    filename
    """
    stocks = tushare.get_stock_basics()

    results = []
    # start date
    start_date = '2019-05-01'

    # end date
    end_date = '2019-06-14'

    count = 0
    for index, row in stocks.iterrows():
        print(row.name)
        result = tushare.get_h_data(code=row.name, index=True, start=start_date, end=end_date)
        if result.empty:
            continue

        print(result)
        print(cal_macd_system(result, 12, 26, 9))
        results.append(result)

        count += 1
        if count == 5:
            break
        time.sleep(20)

    print(results)

    # save data to csv
    results.to_csv('stock.csv', mode='w')
