import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import sys

sys.path.append("..")
from tools import downcast, gen_feat_name
pd.options.mode.chained_assignment = None

import gc

JOBS = 7
CAT_SHIFT = 1


class TimeReversalSimpleDropna:
    def __init__(self, key, primary_time, label, max_win=1, palm_size=-1, palt_list=None, feat_exp=False):
        if palt_list is None:
            palt_list = []
        self.key = key
        self.primary_time = primary_time
        self.label = label
        self.palm_size = palm_size
        self.palt_list = palt_list
        self.feat_exp = feat_exp
        self.record = None
        self.cat2label = {}
        self.max_win = max_win

    def train_fit(self, df):
        df = df[[self.key, self.primary_time, self.label]]
        self.record = df.sort_values(by=[self.key, self.primary_time])

    def test_fit(self, df):
        for val in df[[self.key, self.label]].values:
            cat = val[0]
            if cat in self.cat2label:
                self.cat2label[cat].insert(0, val[1])
                del self.cat2label[cat][self.max_win:]
            else:
                self.cat2label[cat] = [val[1]]

    def reduce_memory(self):
        self.record = self.record[ self.record[self.key]!=self.record[self.key].shift(-self.max_win) ]
        for val in self.record[[self.key, self.label]].values[::-1]:
            cat = val[0]
            if cat in self.cat2label:
                self.cat2label[cat].append(val[1])
            else:
                self.cat2label[cat] = [val[1]]
        self.record = None
        gc.collect()

    def feat_expend(self, df, drop_fe=None):
        if drop_fe is None:
            drop_fe = set()
        if self.feat_exp:
            # 就用一个
            j = 0
            new_col = f'{self.key}_{self.primary_time}_{self.label}_{j+1}_slope'
            new_col = gen_feat_name(self.__class__.__name__, new_col, 'num')
            if new_col not in drop_fe:
                tmp = (df[self.new_cols[j]]-df[self.new_cols[j+1]])/(df[self.new_cols[j+1]] + 1)
                tmp[(tmp == np.inf) | (tmp == -np.inf)] = np.nan
                df[new_col] = tmp.values
            for j in range(min(3, len(self.new_cols)-1)):
                new_col = f'{self.key}_{self.primary_time}_{self.label}_{j + 1}_div'
                new_col = gen_feat_name(self.__class__.__name__, new_col, 'num')
                if new_col not in drop_fe:
                    tmp = df[self.new_cols[j]] / (df[self.new_cols[j + 1]]+1)
                    tmp[(tmp == np.inf) | (tmp == -np.inf)] = np.nan
                    df[new_col] = tmp.values
            j = 0
            new_col = f'{self.key}_{self.primary_time}_{self.label}_{j + 1}_iszero'
            new_col = gen_feat_name(self.__class__.__name__, new_col, 'cat')
            if new_col not in drop_fe:
                df[new_col] = downcast((df[self.new_cols[j]]!=0).astype(int))

        if self.palm_size > 0:
            feat_num = len(self.new_cols) // self.palm_size
            for i in range(feat_num):
                j = i * self.palm_size
                col = self.new_cols[j:j + self.palm_size]
                new_col = f'{self.key}_{self.primary_time}_{self.label}_{j+1}_{j + self.palm_size}'
                new_col = gen_feat_name(self.__class__.__name__, new_col, 'num')
                # ex_cols.append(new_col)
                if new_col not in drop_fe:
                    df[new_col] = df[col].mean(axis=1)

        todo_func = ['mean', 'max', 'min']
        new_col1 = f'{self.key}_{self.primary_time}_{self.label}'
        for j in self.palt_list:
            if j > self.max_win:
                break
            col = self.new_cols[:j]
            new_col2 = f'{new_col1}_{j}'
            for f in todo_func:
                new_col3 = f'{new_col2}_{f}'
                new_col3 = gen_feat_name(self.__class__.__name__, new_col3, 'num')
                if new_col3 not in drop_fe:
                    df[new_col3] = getattr(df[col], f)(axis=1)

    def train_transform(self, X):
        def func(shift):
            new_col = f'{self.key}_{self.primary_time}_{self.label}_{shift}'
            ss = self.record[self.label].shift(shift)
            ss[self.record[self.key] != self.record[self.key].shift(shift)] = np.nan
            ss.name = new_col
            ss = downcast(ss)
            return ss

        df = X if isinstance(X, pd.DataFrame) else X.data
        res = Parallel(n_jobs=JOBS, require='sharedmem')(
            delayed(func)(i) for i in range(1, self.max_win+1))
        self.new_cols = []
        if res:
            res = pd.concat(res, sort=True, axis=1)
            for col in res.columns[:]:
                new_col = gen_feat_name(self.__class__.__name__, col, 'num')
                self.new_cols.append(new_col)
                df[new_col] = res[col].values
        self.reduce_memory()
        self.feat_expend(df)
        return self.new_cols

    def test_transform(self, X, fe=None):
        if fe is None:
            fe = set()
        df = X if isinstance(X, pd.DataFrame) else X.data

        cats = df[self.key].values
        vals = []
        for i in cats:
            if i in self.cat2label:
                vals.append(self.cat2label[i])
            else:
                vals.append([])
        for val in vals:
            if len(val) > 0:
                val.extend([val[-1] for _ in range(len(val), self.max_win)])
                break
        new_cols = [f'{self.key}_{self.primary_time}_{self.label}_{i}' for i in range(1, self.max_win+1)]
        res = pd.DataFrame(vals, columns=new_cols, index=cats, dtype='float32')
        for col in res.columns:
            new_col = gen_feat_name(self.__class__.__name__, col, 'num')
            df[new_col] = downcast(res[col].values)
        self.feat_expend(df, fe)


class TimeReversalSimple:
    def __init__(self, key, primary_time, label, max_win=1, palm_size=-1, palt_list=None, feat_exp=False):
        if palt_list is None:
            palt_list = []
        self.key = key
        self.primary_time = primary_time
        self.label = label
        self.palm_size = palm_size
        self.palt_list = palt_list
        self.feat_exp = feat_exp
        self.record = None
        self.cat2label = {}
        self.max_win = max_win

    def test_fit(self, df):
        for val in df[[self.key, self.label]].values:
            cat = val[0]
            if cat in self.cat2label:
                self.cat2label[cat].insert(0, val[1])
                del self.cat2label[cat][self.max_win:]
            else:
                self.cat2label[cat] = [val[1]]
        na_key = set(self.cat2label.keys()) - set(df[self.key].values)
        for cat in na_key:
            self.cat2label[cat].insert(0, self.cat2label[cat][0])  # self.cat2label[cat][0]
            del self.cat2label[cat][self.max_win:]

    def reduce_memory(self):
        self.record = self.record[ self.record[self.key]!=self.record[self.key].shift(-self.max_win) ]
        for val in self.record[[self.key, self.label]].values[::-1]:
            cat = val[0]
            if cat in self.cat2label:
                self.cat2label[cat].append(val[1])
            else:
                self.cat2label[cat] = [val[1]]
        self.record = None
        gc.collect()

    def feat_expend(self, df, drop_fe=None):

        if drop_fe is None:
            drop_fe = set()
        if self.palm_size > 0:
            feat_num = len(self.new_cols) // self.palm_size
            for i in range(feat_num):
                j = i * self.palm_size
                col = self.new_cols[j:j + self.palm_size]
                new_col = f'{self.key}_{self.primary_time}_{self.label}_{j}_{j + self.palm_size}'
                new_col = gen_feat_name(self.__class__.__name__, new_col, 'num')
                if new_col not in drop_fe:
                    df[new_col] = df[col].mean(axis=1)

        todo_func = ['mean', 'max', 'min']
        new_col1 = f'{self.key}_{self.primary_time}_{self.label}'
        for j in self.palt_list:
            if j > self.max_win:
                break
            col = self.new_cols[:j]
            new_col2 = f'{new_col1}_{j}'
            for f in todo_func:
                new_col3 = f'{new_col2}_{f}'
                new_col3 = gen_feat_name(self.__class__.__name__, new_col3, 'num')
                if new_col3 not in drop_fe:
                    df[new_col3] = getattr(df[col], f)(axis=1)

    def train_transform(self, X):
        def func(shift):
            new_col = f'{self.key}_{self.primary_time}_{self.label}_{shift}'
            ss = self.record[self.label].shift(shift)
            ss[self.record[self.key] != self.record[self.key].shift(shift)] = np.nan
            ss.name = new_col
            ss = downcast(ss)
            return ss

        df = X if isinstance(X, pd.DataFrame) else X.data
        self.record = df[[self.key, self.primary_time, self.label]]
        res = Parallel(n_jobs=JOBS, require='sharedmem')(
            delayed(func)(i) for i in range(1, self.max_win+1))
        self.new_cols = []
        if res:
            res = pd.concat(res, sort=False, axis=1)
            for col in res.columns[:]:
                new_col = gen_feat_name(self.__class__.__name__, col, 'num')
                self.new_cols.append(new_col)
                df[new_col] = res[col].values
        self.reduce_memory()
        self.feat_expend(df)
        return self.new_cols

    def test_transform(self, X, fe=None):
        if fe is None:
            fe = set()
        df = X if isinstance(X, pd.DataFrame) else X.data
        cats = df[self.key].values
        vals = []
        for i in cats:
            if i in self.cat2label:
                vals.append(self.cat2label[i])
            else:
                vals.append([])
        for val in vals:
            if len(val) > 0:
                val.extend([val[-1] for _ in range(len(val), self.max_win)])
                break
        new_cols = [f'{self.key}_{self.primary_time}_{self.label}_{i}' for i in range(1, self.max_win + 1)]
        res = pd.DataFrame(vals, columns=new_cols, index=cats, dtype='float32')
        for col in res.columns:
            new_col = gen_feat_name(self.__class__.__name__, col, 'num')
            df[new_col] = downcast(res[col].values)
        self.feat_expend(df, fe)


class TimeReversalMeanSimpleDropna:
    def __init__(self, key, primary_time, label, max_win=1):
        self.key = key
        self.primary_time = primary_time
        self.label = label
        self.record = None
        self.cat2label = {}
        self.max_win = max_win

    def train_fit(self, df):
        df = df[[self.key, self.primary_time, self.label]]
        self.record = df.groupby([self.key, self.primary_time])[self.label].mean().reset_index()

    def test_fit(self, df):
        df = df.groupby([self.key, self.primary_time])[self.label].mean().reset_index()
        for val in df[[self.key, self.label]].values:
            cat = val[0]
            if cat in self.cat2label:
                self.cat2label[cat].insert(0, val[1])
                del self.cat2label[cat][self.max_win:]
            else:
                self.cat2label[cat] = [val[1]]

    def reduce_memory(self):
        self.record = self.record[ self.record[self.key]!=self.record[self.key].shift(-self.max_win) ]
        for val in self.record[[self.key, self.label]].values[::-1]:
            cat = val[0]
            if cat in self.cat2label:
                self.cat2label[cat].append(val[1])
            else:
                self.cat2label[cat] = [val[1]]
        gc.collect()

    def train_transform(self, X):
        def func(shift):
            new_col = f'{self.key}_{self.primary_time}_{self.label}_{shift}'
            ss = self.record[self.label].shift(shift)
            ss[self.record[self.key] != self.record[self.key].shift(shift)] = np.nan
            ss.name = new_col
            ss = downcast(ss)
            return ss

        df = X if isinstance(X, pd.DataFrame) else X.data
        res = Parallel(n_jobs=JOBS, require='sharedmem')(
            delayed(func)(i) for i in range(1, self.max_win+1))

        new_cols = []
        if res:
            res = pd.concat(res, sort=False, axis=1)
            res[self.primary_time] = self.record[self.primary_time]
            res[self.key] = self.record[self.key]
            tmp = df[[self.primary_time, self.key]]
            tmp = tmp.merge(res, how='left', on=[self.primary_time, self.key])
            for col in tmp.columns[2:]:
                new_col = gen_feat_name(self.__class__.__name__, col, 'num')
                new_cols.append(new_col)
                df[new_col] = tmp[col].values
        self.reduce_memory()
        return new_cols

    def test_transform(self, X):
        df = X if isinstance(X, pd.DataFrame) else X.data
        cats = df[self.key].values
        vals = []
        for i in cats:
            if i in self.cat2label:
                vals.append(self.cat2label[i])
            else:
                vals.append([])
        for val in vals:
            if len(val) > 0:
                val.extend([val[-1] for _ in range(len(val), self.max_win)])
                break
        new_cols = [f'{self.key}_{self.primary_time}_{self.label}_{i}' for i in range(1, self.max_win + 1)]
        res = pd.DataFrame(vals, columns=new_cols, index=cats, dtype='float64')
        for col in res.columns:
            new_col = gen_feat_name(self.__class__.__name__, col, 'num')
            df[new_col] = downcast(res[col].values)


class TimeReversalMeanSimpleNoKey:
    def __init__(self, primary_time, label, max_win=1):
        self.primary_time = primary_time
        self.label = label
        self.record = None
        self.label_list = []
        self.max_win = max_win

    def train_fit(self, df):
        df = df[[self.primary_time, self.label]]
        self.record = df.groupby(self.primary_time)[self.label].mean().reset_index()

    def test_fit(self, df):
        mea = df[self.label].mean()
        self.label_list.insert(0, mea)
        del self.label_list[self.max_win:]

    def reduce_memory(self):
        self.record = self.record.iloc[ -self.max_win:, : ]
        self.label_list = list(self.record[self.label])[::-1]
        gc.collect()

    def train_transform(self, X):
        def func(shift):
            new_col = f'{self.primary_time}_{self.label}_{shift}'
            ss = downcast(self.record[self.label].shift(shift))
            ss.name = new_col
            return ss

        df = X if isinstance(X, pd.DataFrame) else X.data
        res = Parallel(n_jobs=JOBS, require='sharedmem')(
            delayed(func)(i) for i in range(1, self.max_win+1))

        new_cols = []
        if res:
            res = pd.concat(res, sort=False, axis=1)
            res = res.set_index(self.record[self.primary_time])
            tmp = df[[self.primary_time]]
            tmp = tmp.join(res, how='left', on=self.primary_time)
            for col in tmp.columns[1:]:
                new_col = gen_feat_name(self.__class__.__name__, col, 'num')
                new_cols.append(new_col)
                df[new_col] = tmp[col].values
        self.reduce_memory()
        return new_cols

    def test_transform(self, X):
        df = X if isinstance(X, pd.DataFrame) else X.data
        for shift in range(1, self.max_win+1):
            new_col = f'{self.primary_time}_{self.label}_{shift}'
            new_col = gen_feat_name(self.__class__.__name__, new_col, 'num')
            df[new_col] = self.label_list[shift-1]


class PrimaryAggFast:
    def __init__(self, cats):
        self.cats = cats
        self.cat_max = []
        self.unique = []
        self.combine_cat = []
        self.shift = 1

    def train_fit(self, df):
        for cat in self.cats:
            self.cat_max.append( int(df[cat].max()*1.1) )
        for i in self.cat_max:
            self.shift *= (i+1)

    def train_transform(self, X):
        df = X if isinstance(X, pd.DataFrame) else X.data
        category = df[self.cats[0]].astype('float64')
        for i in range(1, len(self.cats)):
            category *= self.cat_max[i]
            category += df[self.cats[i]]
        category[category == (CAT_SHIFT - 1)] = np.nan
        new_col = '_'.join(self.cats) + '_combineID'
        new_col = gen_feat_name(self.__class__.__name__, new_col, 'cat')
        category = downcast(category, accuracy_loss=False)
        df[new_col] = category
        return new_col

    def get_feat_name(self):
        new_col = '_'.join(self.cats) + '_combineID'
        new_col = gen_feat_name(self.__class__.__name__, new_col, 'cat')
        return new_col

    def test_fit(self, df):
        self.judge = (df[self.cats[0]] > self.cat_max[0])
        for i in range(1, len(self.cats)):
            self.judge |= (df[self.cats[i]] > self.cat_max[i])
        df = df[self.judge]
        if not df.empty:
            self.combine_cat = df[self.cats].apply(lambda x: ''.join(map(str, x.values)) if x.any() else np.nan, axis=1)
            unique = self.combine_cat.dropna().drop_duplicates().values
            if len(self.unique) == 0:
                self.unique = sorted(list(unique))
            else:
                added_cats = sorted(set(unique) - set(self.unique))
                self.unique.extend(added_cats)

    def test_transform(self, X):
        df = X if isinstance(X, pd.DataFrame) else X.data
        df_1 = df.loc[~self.judge, :]
        self.train_transform(df_1)
        df_2 = df[self.judge]
        new_col = '_'.join(self.cats) + '_combineID'
        new_col = gen_feat_name(self.__class__.__name__, new_col, 'cat')
        if not df_2.empty:
            codes = pd.Categorical(self.combine_cat, categories=self.unique).codes + CAT_SHIFT + self.shift
            codes = codes.astype('float')
            codes[codes == (CAT_SHIFT - 1)] = np.nan
            codes = downcast(codes, accuracy_loss=False)
            df_2[new_col] = codes
            df_1 = pd.concat([df_1, df_2], sort=False)
            df_1 = df_1.sort_index()
        df[new_col] = df_1[new_col].values

    def fit_transform(self, X, ttype='train'):
        if ttype == 'train':
            self.train_fit(X.data)
            return self.train_transform(X)
        else:
            self.test_fit(X.data)
            self.test_transform(X)


class TimeDate:
    def __init__(self, primary_timestamp):
        self.primary_timestamp = primary_timestamp
        self.attrs = []

    def fit(self, df):
        for atr, nums in zip(['year', 'month', 'day', 'hour', 'weekday', 'minute'], [2, 12, 28, 24, 7, 4]):
            atr_ss = getattr(df[self.primary_timestamp].dt, atr)
            if atr_ss.nunique() >= nums:
                self.attrs.append(atr)

    def transform(self, X):
        df = X if isinstance(X, pd.DataFrame) else X.data
        for atr in self.attrs:
            new_col = self.primary_timestamp + '_' + atr
            df[new_col] = downcast(getattr(df[self.primary_timestamp].dt, atr), accuracy_loss=False)
        df[self.primary_timestamp] = df[self.primary_timestamp].astype('int64') // 10 ** 9

    def fit_transform(self, X, ttype):
        if ttype == 'train':
            self.fit(X.data)
            self.transform(X)
        else:
            self.transform(X)
