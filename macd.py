#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tushare
import pandas as pd
import time
import sqlite3


def getstartdate(code):
    """obtain establish date of stock"""
    df = tushare.get_stock_basics()
    date = df.iloc[code]['timeToMarket']  # 上市日期YYYYMMDD
    return date


def get_EMA(df, N):
    # print(df)
    # df['ema'] = None
    ema = [0] * len(df)

    for i in range(len(df)):
        if i == 0:
            ema[i] = df.iloc[i]['close']
        else:
            ema[i] = (2 * df.iloc[i]['close'] + (N - 1) * ema[i - 1]) / (N + 1)

    return ema


def get_MACD(df, short=12, long=26, M=9):
    a = get_EMA(df, short)
    b = get_EMA(df, long)
    ema = pd.DataFrame()
    ema['a'] = a
    ema['b'] = b

    df['diff'] = ema['a'] - ema['b']
    # print(df)
    dea = [0] * len(df)

    for i in range(len(df)):
        if i == 0:
            dea[i] = df.iloc[i]['diff']
        else:
            dea[i] = (2 * df.iloc[i]['diff'] + (M - 1) * dea[i - 1]) / (M + 1)

    df['dea'] = dea

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
    conn = sqlite3.connect('stock.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE STOCK
           (CODE        VARCHAR(8)      NOT NULL,
            DATE        DATE            NOT NULL,
            OPEN        REAL,
            HIGH        REAL,
            CLOSE       REAL,
            LOW         REAL,
            VOLUME      BIGINT,
            AMOUNT      BIGINT,
            DIFF        REAL,
            DEA         REAL,
            MACD        REAL);''')
    conn.commit()

    conn.close()

    stocks = tushare.get_stock_basics()

    results = None
    # start date
    start_date = '2018-01-01'

    # end date
    end_date = '2019-06-14'

    count = 0
    for code, row in stocks.iterrows():
        print(row.name)
        result = tushare.get_h_data(code=code, index=True, start=start_date, end=end_date)
        if result.empty:
            continue

        result = result.sort_index()
        result['code'] = code
        result = result.reset_index()
        get_MACD(result, 12, 26, 9)
        print(result)
        results = pd.concat([results, result], ignore_index=True)

        result.to_csv('stock.csv', mode='a', header=count == 0, index=False)

        count += 1
        if count == 3:
            break
        # time.sleep(20)

    print(results)

    # save data to csv
    # results.to_csv('stock.csv', mode='w')
