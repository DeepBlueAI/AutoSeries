
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
import math


def downcast(series, accuracy_loss=True, min_float_type='float16'):
    if series.dtype == np.int64:
        ii8 = np.iinfo(np.int8)
        ii16 = np.iinfo(np.int16)
        ii32 = np.iinfo(np.int32)
        max_value = series.max()
        min_value = series.min()

        if max_value <= ii8.max and min_value >= ii8.min:
            return series.astype(np.int8)
        elif max_value <= ii16.max and min_value >= ii16.min:
            return series.astype(np.int16)
        elif max_value <= ii32.max and min_value >= ii32.min:
            return series.astype(np.int32)
        else:
            return series

    elif series.dtype == np.float64:
        fi16 = np.finfo(np.float16)
        fi32 = np.finfo(np.float32)

        if accuracy_loss:
            max_value = series.max()*7
            min_value = series.min()*7
            if np.isnan(max_value):
                max_value = 0

            if np.isnan(min_value):
                min_value = 0

            if min_float_type == 'float16' and max_value <= fi16.max and min_value >= fi16.min:
                return series.astype(np.float16)
            elif max_value <= fi32.max and min_value >= fi32.min:
                return series.astype(np.float32)
            else:
                return series
        else:
            tmp = series[~pd.isna(series)]
            if len(tmp) == 0:
                return series.astype(np.float16)

            if (tmp == tmp.astype(np.float16)).sum() == len(tmp):
                return series.astype(np.float16)
            elif (tmp == tmp.astype(np.float32)).sum() == len(tmp):
                return series.astype(np.float32)

            else:
                return series

    else:
        return series


def gen_feat_name(cls_name, feat_name, feat_type):
    prefix = feat_type
    return f"({prefix}){cls_name}:{feat_name}"


def gen_combine_cats(df, cols):
    category = df[cols[0]].astype('float64')
    for col in cols[1:]:
        mx = df[col].max()
        category *= mx
        category += df[col]
    return category


def time_train_test_split(X, y, primary_time, test_rate=0.2, shuffle=False, random_state=1, jia_y=True):
    if jia_y:
        changed_y = X.pop('changed_y')
    timelist = list(X[primary_time].unique())
    length = len(timelist)

    test_size = int(length * test_rate)
    train_size = length - test_size
    train_time = timelist[:train_size]
    test_time = timelist[train_size:]

    X_train = X[X[primary_time].isin(train_time)]
    X_test = X[X[primary_time].isin(test_time)]

    train_num = X_train.shape[0]
    y_train = y.iloc[:train_num]
    if jia_y:
        y_test = changed_y.iloc[train_num:]
    else:
        y_test = y.iloc[train_num:]

    if shuffle:
        np.random.seed(random_state)
        idx = np.arange(train_size)
        np.random.shuffle(idx)
        X_train = X_train.iloc[idx]
        y_train = y_train.iloc[idx]
    return X_train, y_train, X_test, y_test


def data_sample_bykey_rate(train_data, key, agg_key, primary_time):
    df_raw_num = train_data.shape[0]
    len2frac = [(1000000, 0.15), (100000, 0.2), (30000, 0.3), (0, 0.6)]
    frac = 0.6
    for df_len, f in len2frac:
        if df_raw_num > df_len:
            frac = f
            break
    time_list = list(train_data[primary_time])
    time_list = time_list[int(-len(time_list) * frac):]
    if len(time_list) > 200:
        train_data = train_data[train_data[primary_time].isin(time_list)]
        return train_data.reset_index(drop=True)
    nun = train_data[key].nunique()
    key = nun.idxmax()
    for df_len, frac in len2frac:
        if df_raw_num > df_len:
            if len(key) == 0:
                train_data = train_data.sample(frac=frac)
            else:
                primary_list = pd.Series(train_data[key].unique())
                primary_list = primary_list.sample(frac=frac)
                if len(primary_list) > 2:
                    train_data = train_data[train_data[key].isin(primary_list.values)]
                else:
                    primary_list = pd.Series(train_data[agg_key].unique())
                    sample_primary = primary_list.sample(frac=frac)
                    if len(sample_primary) > 0:
                        train_data = train_data[train_data[agg_key].isin(sample_primary.values)]
                    else:
                        train_data = train_data.sample(frac=frac)
            break
    return train_data.reset_index(drop=True)


def sample_bykey_num(X_train, y_train, key, agg_key, primary_time):
    df_raw_num = X_train.shape[0]
    frac = min(1, 50000 / df_raw_num)
    time_list = list(X_train[primary_time])
    time_list = time_list[int(-len(time_list) * frac):]
    if len(time_list) > 160:
        X_train = X_train[X_train[primary_time].isin(time_list)]
        ind = X_train.index
        y_train = y_train[ind]
        return X_train, y_train

    nun = X_train[key].nunique()
    key = nun.idxmax()
    if len(key) == 0:
        X_train = X_train.sample(frac=frac)
    else:
        primary_list = pd.Series(X_train[key].unique())
        primary_list = primary_list.sample(frac=frac)
        if len(primary_list) > 10:
            X_train = X_train[X_train[key].isin(primary_list.values)]
        else:
            primary_list = pd.Series(X_train[agg_key].unique())
            sample_primary = primary_list.sample(frac=frac)
            if len(sample_primary) > 10:
                X_train = X_train[X_train[agg_key].isin(sample_primary.values)]
            else:
                X_train = X_train.sample(frac=frac)

    index = X_train.index
    y_train = y_train[index]
    return X_train, y_train


def rmse(preds, real):
    return math.sqrt(mean_squared_error(real, preds))


def serch_best_fusion_proportion(predsa, predsb, real):
    weight_b = 0
    score_min = rmse(predsa, real)
    last_score = score_min
    direc = 0

    w = 0
    while w <= 1:
        w += 0.05
        preds = predsa * (1-w) + predsb * w

        score = rmse(preds, real)
        if score < score_min:
            score_min = score
            weight_b = w

        if direc * (score - last_score) < 0:
            break
        direc = score - last_score
        last_score = score

    return 1 - weight_b, weight_b
