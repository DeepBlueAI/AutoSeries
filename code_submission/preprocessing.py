import gc
import pandas as pd
import numpy as np

from default_feat import PrimaryAggFast, TimeDate
from logger_control import time_limit
from tools import downcast

CAT_SHIFT = 1


class Preprocess:
    def train_preprocess(self, X):
        self.catPreprocessor = CatPreprocessor(X.dtype_cols['cat'])
        self.catPreprocessor.fit_transform(X.data)

        self.timeDate = TimeDate(X.primary_timestamp)
        self.timeDate.fit_transform(X, 'train')

        with time_limit("PrimaryAgg"):
            if len(X.primary_id) > 1:
                self.primaryAgg = PrimaryAggFast(X.primary_id)
                self.primaryAgg.train_fit(X.data)
                col_name = self.primaryAgg.train_transform(X)
                X.primary_agg = col_name
            elif len(X.primary_id) == 0:
                X.data['PrimaryAgg_same'] = 1
                X.primary_agg = 'PrimaryAgg_same'
            elif len(X.primary_id) == 1:
                X.primary_agg = X.primary_id[0]

        with time_limit('AbnormalTrainLabel'):
            self.ab = AbnormalTrainLabel(X.primary_agg, X.label)
            self.ab.fit_transform(X, ttype='train')

    def test_preprocess(self, X):

        self.catPreprocessor.fit_transform(X.data)

        self.timeDate.fit_transform(X, 'test')

        if len(X.primary_id) > 1:
            self.primaryAgg.test_fit(X.data)
            self.primaryAgg.test_transform(X)
        elif len(X.primary_id) == 0:
            X.data['PrimaryAgg_same'] = 1

        self.ab.fit_transform(X, ttype='test')


class CatPreprocessor:
    def __init__(self, cats):
        self.cats = cats
        self.cats2unique = {}

    def fit(self, ss):
        for cat in self.cats:
            unique = ss[cat].dropna().drop_duplicates().values
            if cat not in self.cats2unique:
                self.cats2unique[cat] = sorted(list(unique))
            else:
                added_cats = sorted(set(unique) - set(self.cats2unique[cat]))
                self.cats2unique[cat].extend(added_cats)

    def transform(self, ss):
        for cat in self.cats:
            codes = pd.Categorical(ss[cat], categories=self.cats2unique[cat]).codes + CAT_SHIFT
            codes = codes.astype('float')
            codes = downcast(codes, accuracy_loss=False)
            ss[cat] = codes

    def fit_transform(self, ss):
        self.fit(ss)
        self.transform(ss)


class DataFormat:
    def __init__(self, primary_timestamp, primary_id):
        self.primary_timestamp = primary_timestamp
        self.primary_id = primary_id

    def fit(self, df):
        pass

    def transform(self, df):
        df[self.primary_timestamp] = pd.to_datetime(df[self.primary_timestamp], unit='s')

    def fit_transform(self, df):
        self.fit(df)
        self.transform(df)


class TypeAdapter:
    def __init__(self, primitive_cat):
        self.adapt_cols = primitive_cat.copy()

    def fit_transform(self, X):
        cols_dtype = dict(zip(X.columns, X.dtypes))

        for key, dtype in cols_dtype.items():
            if dtype == np.dtype('object'):
                self.adapt_cols.append(key)
            if key in self.adapt_cols:
                X[key] = X[key].apply(hash_m)

    def transform(self, X):
        for key in X.columns:
            if key in self.adapt_cols:
                X[key] = X[key].apply(hash_m)


def hash_m(x):
    return hash(x) % 1048575


def time_interval(ss):
    ss = ss.iloc[int(ss.shape[0]*0.7):]
    ss = ss[ss.diff() != 0]
    min_interval = ss.diff().min()
    ss = pd.to_datetime(ss, unit='s')
    year = ss.dt.year
    if year.nunique() == len(year):
        tt = min_interval//(3600*24*365-2)
        cnt = (year.max()-year.min())/tt - len(ss-1)
        return tt, 'y', int(cnt/0.7)

    month = ss.dt.month
    diff = month.diff()
    if (diff==0).sum() == 0:
        tt = min_interval // (3600 * 24 * 28 - 2)
        diff[diff < 0] += 12
        cnt = diff.sum()/tt - (len(ss)-1)
        return tt, 'm', int(cnt/0.7)

    day = ss.dt.day
    diff = day.diff()
    if (diff == 0).sum() == 0:
        tt = min_interval // (3600 * 24 - 2)
        diff[diff < 0] = tt
        cnt = diff.sum()/tt - (len(ss)-1)
        if tt == 7:
            return tt, 'w', int(cnt/0.7)
        return tt, 'd', int(cnt/0.7)

    hour = ss.dt.hour
    diff = hour.diff()
    if (diff == 0).sum() == 0:
        tt = min_interval // (3600 - 1)
        diff[diff < 0] += 24
        cnt = diff.sum()/tt - (len(ss)-1)
        return tt, 'h', int(cnt/0.7)

    minute = ss.dt.minute
    diff = minute.diff()
    if (diff == 0).sum() == 0:
        tt = min_interval // (60 - 1)
        diff[diff < 0] += 60
        cnt = diff.sum()/tt - (len(ss) - 1)
        return tt, 'M', int(cnt/0.7)

    second = ss.dt.second
    diff = second.diff()
    tt = min_interval
    diff[diff < 0] += 60
    cnt = diff.sum()/tt - (len(ss) - 1)
    return tt, 's', int(cnt/0.7)


class AbnormalTrainLabel:
    def __init__(self, key, label):
        self.key = key
        self.label = label
        self.cat2lable1 = {}
        self.cat2lable2 = {}
        self.max_val = None
        self.min_val = None

    def train_transform(self, X):
        df = X.data

        group = df.groupby(X.primary_agg)[self.label]
        cat2mean = group.mean()
        cat2std = group.std(ddof=1)
        self.max_val = cat2mean + cat2std*6
        self.min_val = cat2mean - cat2std*6

        def func(ss):
            name = ss.name
            ss = ss.values
            ls = len(ss)
            m = cat2mean[name]
            max_v = self.max_val[name]
            min_v = self.min_val[name]
            for i in range(ls):
                ll = ss[i-1] if i > 0 else m
                rr = ss[i+1] if i < ls-1 else m
                a = (ll+rr)/2
                up = abs(ss[i]-a)
                down = abs(ll-rr)+1
                frac = up/down
                if (frac > 10) and (up>abs(a)*10) and ((max_v<ss[i]) or (ss[i] < min_v)):
                    ss[i] = a + up/((frac/10)**1.8)

            return ss

        df[self.label] = group.transform(func)

    def test_fit(self, df):
        df = df[[self.key, self.label]].values
        for cat, lab in df:
            if cat in self.cat2lable1:
                self.cat2lable2[cat] = self.cat2lable1[cat]
            self.cat2lable1[cat] = lab

    def test_transform(self, df):
        val = df[[self.key, self.label]].values
        lab_list = []
        for cat, lab in val:
            if cat not in self.cat2lable2:
                lab_list.append(lab)
                continue
            ll = self.cat2lable2[cat]
            rr = self.cat2lable1[cat]
            a = (ll+rr) / 2
            up = abs(lab - a)
            down = abs(ll-rr) + 1
            frac = up / down
            if cat not in self.max_val:
                self.max_val[cat] = self.max_val.mean()
                self.min_val[cat] = self.min_val.mean()
            max_v = self.max_val[cat]
            min_v = self.min_val[cat]
            if (frac > 10) and (up > abs(a) * 10) and ((max_v<lab) or (lab < min_v)):
                lab = a + up / ((frac / 10) ** 1.8)
            lab_list.append(lab)

    def fit_transform(self, X, ttype):
        if ttype=='train':
            self.train_transform(X)
        else:
            if not X.isfirst_predict:
                self.test_transform(X.history)
                self.test_fit(X.history)


class TransExponentialDecay:
    def __init__(self, primary_timestamp, init=1.0, finish=0.5, offset=0):
        self.primary_timestamp = primary_timestamp
        self.init = init
        self.finish = finish
        self.offset = offset

    def get_decayed_value(self, t):
        if t < self.offset:
            return self.init
        return 2 - np.exp(-self.alpha * (t - self.offset + self.l))

    def fit(self, df):
        time_col = df[self.primary_timestamp]
        unique_time = sorted(time_col.unique(), reverse=True)
        self.alpha = np.log(-self.init / (self.finish - 2)) / len(unique_time)
        self.l = -np.log(self.init) / self.alpha
        weight = []
        for i in range(1, len(unique_time)+1):
            weight.append(self.get_decayed_value(i))

        weight_map = {}
        for k, w in zip(unique_time, weight):
            weight_map[k] = w

        time_col = time_col.map(weight_map)
        df['sample_weight'] = time_col
