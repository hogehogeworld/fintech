#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: set fileencoding=utf-8 :
#
# Author:   Ashigirl96
# URL:      http://pwn.hatenablog.com/
# License:  MIT License
# Created:  2016-06-08

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import statistics as st
import pandas as pd
import pandas.io.data as web
from arch import arch_model

start = '1971-01-04'

# data['Adj Close'].head().pct_change().dropna()

jpy = web.DataReader('DEXJPUS', 'fred', start=start)


print(jpy['DEXJPUS'].head().pct_change().dropna())
print("*"*60)
ret = jpy['DEXJPUS'].pct_change().dropna()


## GARCH(1, 1)
am = arch_model(ret)
res = am.fit(update_freq=5)
print(res.summary())

print("*"*60)

## GJR-GARCH
start = '2010-01-01'
jpy = web.DataReader('DEXJPUS', 'fred', start=start)
ret = jpy['DEXJPUS'].pct_change().dropna()
gjr_am = arch_model(ret, p=1, o=1, q=1)
res = gjr_am.fit(update_freq=5, disp='off')
print(res.summary())

print("*"*60)

## TARCH
t_am = arch_model(ret, p=1, o=1, q=1, power=1.0)
res = t_am.fit(update_freq=5)
print(res.summary())
