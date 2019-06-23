#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from datetime import datetime

import tushare
import pandas as pd
import time
import sqlite3

from sqlalchemy import Column, String, create_engine, Integer, Date, Float
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base


def getstartdate(code):
    """obtain establish date of stock"""
    df = tushare.get_stock_basics()
    date = df.iloc[code]['timeToMarket']  # 上市日期YYYYMMDD
    return date


def get_EMA(df, N, ema0):
    ema = [0] * len(df)

    for i in range(len(df)):
        if i == 0:
            if ema0 is None:
                ema[i] = df.iloc[i]['close']
            else:
                ema[i] = (2 * df.iloc[i]['close'] + (N - 1) * ema0) / (N + 1)
        else:
            ema[i] = (2 * df.iloc[i]['close'] + (N - 1) * ema[i - 1]) / (N + 1)

    return ema


def get_MACD(df, short=12, long=26, M=9, ema_short0=None, ema_long0=None, dea0=None):
    ema_short = get_EMA(df, short, ema_short0)
    ema_long = get_EMA(df, long, ema_long0)
    ema = pd.DataFrame()
    df['ema_short'] = ema_short
    df['ema_long'] = ema_long

    df['diff'] = df['ema_short'] - df['ema_long']
    # print(df)
    dea = [0] * len(df)

    for i in range(len(df)):
        if i == 0:
            if dea0 is None:
                dea[i] = df.iloc[i]['diff']
            else:
                dea[i] = (2 * df.iloc[i]['diff'] + (M - 1) * dea0) / (M + 1)
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


def endtime():
    ticks = time.time()

    localtime = time.localtime(ticks - 24 * 60 * 60)
    # end date
    return time.strftime("%Y-%m-%d", localtime)


Base = declarative_base()


class Stock(Base):
    # 指定本类映射到users表
    __tablename__ = 'stock'

    # 指定id映射到id字段; id字段为整型，为主键
    code = Column(String(8), primary_key=True)
    # 指定name映射到name字段; name字段为字符串类形，
    date = Column(Date)
    ema_short = Column(Float)
    ema_long = Column(Float)
    dea = Column(Float)

    def __repr__(self):
        return "<Stock(code=%s, date=%s)>" % (self.code, self.date)


one_day = 24 * 60 * 60

conn = sqlite3.connect('stock.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS STOCK
       (CODE        VARCHAR(8)      NOT NULL,
        DATE        DATE            NOT NULL,
        OPEN        REAL,
        HIGH        REAL,
        CLOSE       REAL,
        LOW         REAL,
        VOLUME      BIGINT,
        AMOUNT      BIGINT,
        EMA_SHORT   REAL,
        EMA_LONG    REAL,
        DIFF        REAL,
        DEA         REAL,
        MACD        REAL);''')
conn.commit()

conn.close()

# 初始化数据库连接:
engine = create_engine('sqlite:///stock.db')

stocks = tushare.get_stock_basics()

end_ticks = time.time()
end_ticks = end_ticks / one_day * one_day - one_day
localtime = time.localtime(end_ticks)
# end date
end_date = time.strftime("%Y-%m-%d", localtime)

# engine是2.2中创建的连接
Session = sessionmaker(bind=engine)

# 创建Session类实例
session = Session()

print(time.strftime("%H:%M:%S", time.localtime()))

count = 0
for code, row in stocks.iterrows():
    print(row.name)

    max_date = session.query(Stock).filter_by(code=code).order_by(Stock.date.desc()).first()

    start_date = '2019-01-01'
    if max_date:
        print(max_date)
        begin_ticks = datetime.strptime(str(max_date.date), '%Y-%m-%d').timestamp()
        begin_ticks += one_day
        if begin_ticks > end_ticks:
            continue

        localtime = time.localtime(begin_ticks)
        start_date = time.strftime("%Y-%m-%d", localtime)

    while True:
        try:
            result = tushare.get_h_data(code=code, index=True, start=start_date, end=end_date)
            break
        except IOError as e:
            print(e)
            time.sleep(10 * 60)
            print(time.strftime("%H:%M:%S", time.localtime()))
    if result.empty:
        continue

    result = result.sort_index()
    result['code'] = code
    result = result.reset_index()

    if max_date:
        get_MACD(result, 9, 13, 8, max_date.ema_short, max_date.ema_long, max_date.dea)
    else:
        get_MACD(result, 9, 13, 8)

    print(result)

    # result.to_csv('stock.csv', mode='a', header=count == 0, index=False)
    result.to_sql('stock', con=engine, if_exists='append', index=False)

    count += 1
    # if count == 3:
    #     break
    # time.sleep(20)

# print(results)

# save data to csv
# results.to_csv('stock.csv', mode='w')
session.close()
