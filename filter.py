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

Base = declarative_base()


class Stock(Base):
    # 指定本类映射到users表
    __tablename__ = 'stock'
    code = Column(String(8))
    date = Column(Date)
    macd = Column(Float)

    __mapper_args__ = {
        'primary_key': [code, date]
    }

    def __repr__(self):
        return "<Stock(code=%s, date=%s,macd=%s)>" % (self.code, self.date, self.macd)


# 初始化数据库连接:
engine = create_engine('sqlite:///stock.db')

stocks = tushare.get_stock_basics()

# engine是2.2中创建的连接
Session = sessionmaker(bind=engine)

# 创建Session类实例
session = Session()

for code, row in stocks.iterrows():
    # print(row.name)
    data = session.query(Stock).filter_by(code=code).order_by(Stock.date.desc()).limit(7).all()
    if len(data):
        # print(data)

        first = data[0].macd
        lowest = True
        for item in data:
            if first > item.macd:
                lowest = False
                break
        if lowest:
            print(data)

session.close()
