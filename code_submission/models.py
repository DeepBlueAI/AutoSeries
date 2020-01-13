import gc
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_squared_error
import math

import CONSTANT
from logger_control import log
from tools import sample_bykey_num


class LinearRegressor:
    def __init__(self):
        self.model = linear_model.LinearRegression(fit_intercept=False)
        self.feature_selector = SelectPercentile(f_regression, percentile=100)
        self.imputer = Imputer(missing_values='NaN', strategy='mean', axis=0, verbose=0, copy=True)
        self.best_columns = []
        self.feature_name = []

    def valid_fit(self, X_train_init, y_train, X_eval_init, y_eval, use_sample_weight=False, use_sample_window_select=False):
        max_sample_num = 1000000
        if len(X_train_init) > max_sample_num:
            X_train_init = X_train_init[-max_sample_num:]
            y_train = y_train[-max_sample_num:]
            gc.collect()

        weight_column = 'sample_weight'
        train_weight = None
        if use_sample_weight:
            train_weight = X_train_init.pop(weight_column)
            eval_weight = X_eval_init.pop(weight_column)

        init_columns = X_train_init.columns
        X_train = self.imputer.fit_transform(X_train_init)
        X_eval = self.imputer.transform(X_eval_init)

        if X_train.shape[1] != len(init_columns):
            X_train_init[(X_train_init == np.inf) | (X_train_init == -np.inf)] = np.nan
            X_train = self.imputer.fit_transform(X_train_init)
            X_eval = self.imputer.transform(X_eval_init)

        score_min = float("inf")
        best_percentile = 100
        best_preds = None
        best_column_num = 0

        if X_train.shape[1] < 20:
            if use_sample_weight:
                self.model.fit(X_train, y_train, sample_weight=train_weight)
            else:
                self.model.fit(X_train, y_train)
            best_preds = self.model.predict(X_eval)
        else:
            for percentile in range(100, 10, -10):
                self.feature_selector.set_params(**{'percentile': percentile})
                gc.collect()
                train = self.feature_selector.fit_transform(X_train, y_train)
                eval = self.feature_selector.transform(X_eval)

                if use_sample_weight:
                    self.model.fit(train, y_train, sample_weight=train_weight)
                else:
                    self.model.fit(train, y_train)

                preds = self.model.predict(eval)
                score = math.sqrt(mean_squared_error(y_eval, preds))
                print(f"valid score:{score}\n")

                if score < score_min:
                    score_min = score
                    best_percentile = percentile
                    best_preds = preds
                    best_column_num = train.shape[1]

                gc.collect()

            ss = pd.Series(self.feature_selector.scores_, index=init_columns)
            score_sorted_cols = list(ss.sort_values(ascending=False).index)
            self.best_columns = score_sorted_cols[:best_column_num]
        return best_preds

    def fit(self, X_train, y_train, use_sample_weight):
        weight_column = 'sample_weight'
        train_weight = None

        if use_sample_weight:
            train_weight = X_train.pop(weight_column)

        if len(self.best_columns):
            self.feature_name = list(set(self.best_columns) & set(X_train.columns))
            X_train = X_train[self.feature_name]
        else:
            self.feature_name = X_train.columns

        init_column_num = len(X_train.columns)
        train = self.imputer.fit_transform(X_train)
        if train.shape[1] != init_column_num:
            X_train[(X_train == np.inf) | (X_train == -np.inf)] = np.nan
            train = self.imputer.fit_transform(X_train)

        if use_sample_weight:
            self.model.fit(train, y_train, sample_weight=train_weight)
        else:
            self.model.fit(train, y_train)

        gc.collect()
        return self

    def predict(self, x_test):
        x_test = x_test[self.feature_name]
        test = self.imputer.transform(x_test)
        pred = self.model.predict(test)
        return pred


class LGBMRegressor:
    def __init__(self, params=None):
        self.model = None

        self.params = {
            'params': {"objective": "regression", "metric": "rmse", 'verbosity': -1, "seed": 0, 'two_round': False,
                       'num_leaves': 20, 'bagging_fraction': 0.9, 'bagging_freq': 3,
                       'feature_fraction': 0.9, 'min_sum_hessian_in_leaf': 0.1, 'lambda_l1': 0.5,
                       'lambda_l2': 0.5, 'min_data_in_leaf': 50
                       },

        }
        self.params1 = {
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": "rmse",
            "verbosity": 1,
            "seed": CONSTANT.SEED,
            "num_threads": CONSTANT.THREAD_NUM
        }

        self.hyperparams = {
            'two_round': False,
            'num_leaves': 20, 'bagging_fraction': 0.9, 'bagging_freq': 3,
            'feature_fraction': 0.9, 'min_sum_hessian_in_leaf': 0.1, 'lambda_l1': 0.5,
            'lambda_l2': 0.5, 'min_data_in_leaf': 50
        }

        self.learning_rates = 0.05
        self.num_boost_round = 400
        self.best_iteration = 0
        self.train_size = 0
        if params is not None:
            self.params = params

    def subsample_opt(self, num_samples):
        samples = num_samples
        if samples > 1000000:
            samples = 1000000

        if samples < 200000:
            subsample = 0.9 - (samples / 1000000)**1.5
            return subsample

        subsample = 0.85 - samples / 2500000
        return subsample

    def valid_fit(self, X_train, y_train, X_eval, y_eval, categorical_feature=None, use_sample_weight=False, round=-1):
        pian = 0
        if round == -1:
            round = self.num_boost_round-100
            pian = 100

        weight_column = 'sample_weight'
        if use_sample_weight:
            train_weight = X_train.pop(weight_column).values
        else:
            train_weight = None
        feat_name_cols = X_train.columns
        self.feature_name = list(X_train.columns)
        feat_name_maps = {feat_name_cols[i]: str(i) for i in range(len(feat_name_cols))}
        new_feat_name_cols = [feat_name_maps[i] for i in feat_name_cols]
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.float32)
        gc.collect()
        X_train = X_train.values
        y_train = y_train.values
        gc.collect()
        X_eval = X_eval.astype(np.float32)
        y_eval = y_eval.astype(np.float32)
        gc.collect()
        X_eval = X_eval.values
        y_eval = y_eval.values
        gc.collect()

        if use_sample_weight:
            lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=feat_name_cols, weight=train_weight)
        else:
            lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=feat_name_cols)
        lgb_eval = lgb.Dataset(X_eval, label=y_eval, feature_name=feat_name_cols, reference=lgb_train)
        self.model = lgb.train(train_set=lgb_train, valid_sets=[lgb_train, lgb_eval],
                               categorical_feature=categorical_feature,
                               feature_name=new_feat_name_cols,
                               init_model=self.model,
                               valid_names=['train', 'eval'], **self.params,
                               num_boost_round=round,
                               early_stopping_rounds=35,
                               verbose_eval=20,
                               learning_rates=self.learning_rates[pian:round+pian])

        self.best_iteration = self.model.best_iteration
        return self.model.predict(X_eval), self.model.best_score["eval"][self.params['params']["metric"]]

    def fit(self, X_train, y_train, categorical_feature=None, use_sample_weight=False):

        self.feature_name = list(X_train.columns)
        weight_column = 'sample_weight'
        if use_sample_weight:
            train_weight = X_train.pop(weight_column).values
        else:
            train_weight = None

        feat_name_cols = list(X_train.columns)
        feat_name_maps = {feat_name_cols[i]: str(i) for i in range(len(feat_name_cols))}
        f_feat_name_maps = {str(i): feat_name_cols[i] for i in range(len(feat_name_cols))}
        new_feat_name_cols = [feat_name_maps[i] for i in feat_name_cols]
        X_train.columns = new_feat_name_cols
        categories = [feat_name_maps[i] for i in categorical_feature]
        self.f_feat_name_maps = f_feat_name_maps
        self.new_feat_name_cols = new_feat_name_cols

        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.float32)
        gc.collect()
        X_train = X_train.values
        y_train = y_train.values
        gc.collect()

        if use_sample_weight:
            lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=feat_name_cols, weight=train_weight)
        else:
            lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=feat_name_cols)

        params = self.params1
        hyperparams = self.hyperparams
        if self.best_iteration > self.num_boost_round:
            self.num_boost_round += 50
            self.get_log_lr(self.num_boost_round, max_lr=self.max_lrs, min_lr=self.min_lrs)
        self.best_iteration = int(1.02*self.best_iteration)
        self.model = lgb.train(train_set=lgb_train, valid_sets=lgb_train, categorical_feature=categorical_feature,
                               valid_names='train', **self.params, num_boost_round=self.best_iteration,
                               early_stopping_rounds=35,
                               verbose_eval=20,
                               )

        return self

    def get_log_lr(self, num_boost_round, max_lr, min_lr):
        learning_rates = [max_lr + (min_lr - max_lr) / np.log(num_boost_round) * np.log(i) for i in
                          range(1, num_boost_round + 1)]
        return learning_rates

    def lr_opt(self, train_data, valid_data, categories):
        params = self.params1
        hyperparams = self.hyperparams
        max_num_boost_round = 200
        raw_nums = [1000000, 500000, 10000, 0]
        num_boost_rounds = [230, 300, 350, 400]
        max_num_boost_rounds = [180, 240, 280, 310]
        for i in range(len(raw_nums)):
            if raw_nums[i] < self.train_size:
                max_num_boost_round = max_num_boost_rounds[i]
                self.num_boost_round = num_boost_rounds[i]
                break

        max_lrs = [0.1, 0.075, 0.05, 0.025]
        min_lrs = [0.05, 0.0325, 0.025, 0.01]

        scores = []
        lrs = []
        for max_lr, min_lr in zip(max_lrs, min_lrs):
            learning_rates = self.get_log_lr(self.num_boost_round, max_lr, min_lr)

            model = lgb.train({**params, **hyperparams}, train_data, num_boost_round=max_num_boost_round,
                              categorical_feature=categories, learning_rates=learning_rates[:max_num_boost_round],
                              )
            pred = model.predict(valid_data.data)
            score = mean_squared_error(valid_data.label, pred)
            scores.append(score)
            lrs.append(learning_rates)
            del model, pred
            gc.collect()

        best_loop = np.argmin(scores)
        self.max_lrs = max_lrs[best_loop]
        self.min_lrs = min_lrs[best_loop]
        best_score = np.min(scores)
        lr = lrs[best_loop]
        log(f'scores {scores}')
        log(f'loop {best_loop}')
        log(f'lr max {lr[0]} min {lr[-1]}')
        log(f'lr best score {best_score}')
        return lr

    def param_opt_new(self, X_train, y_train, X_valid, y_valid, categories, primary_id, primary_agg, primary_time):
        self.train_size = X_train.shape[0] + X_valid.shape[0]
        bagging_fraction = self.subsample_opt(self.train_size)
        self.params['params']['bagging_fraction'] = bagging_fraction
        self.hyperparams['bagging_fraction'] = bagging_fraction
        X, y = sample_bykey_num(X_train, y_train, primary_id, primary_agg, primary_time)
        X.drop(primary_time, axis=1, inplace=True)
        X_valid.drop(primary_time, axis=1, inplace=True)
        feat_name = list(X.columns)
        X = X.astype(np.float32)
        gc.collect()
        y = y.astype(np.float32)
        gc.collect()
        X = X.values
        gc.collect()
        y = y.values
        gc.collect()

        train_data = lgb.Dataset(X, label=y, feature_name=feat_name)
        del X, y
        gc.collect()

        valid_data = lgb.Dataset(X_valid, label=y_valid, feature_name=feat_name, free_raw_data=False)
        del X_valid, y_valid
        gc.collect()

        lr = self.lr_opt(train_data, valid_data, categories)
        del train_data
        gc.collect()
        self.learning_rates = lr
        log(f'pass round opt, use best iteration as {self.best_iteration}')

    def predict(self, X_test):
        X_test = X_test[self.feature_name]
        X_test.columns = self.new_feat_name_cols
        return self.model.predict(X_test)

    def score(self):
        df_imp = pd.DataFrame({'features': [i for i in self.model.feature_name()],
                               'importances': self.model.feature_importance()})

        df_imp.sort_values('importances', ascending=False, inplace=True)
        return df_imp

    def train_set_adjust(self, X_train, y_train, X_eval, y_eval, categorical_feature, primary_time):
        test_time_num = X_eval[primary_time].nunique()
        train_time_list = X_train[primary_time].unique()
        next = True
        min_score = -1
        best_train_time_num = -1
        for i in range(1, 5):
            if not next:
                break
            train_time_num = i * test_time_num
            if train_time_num > len(train_time_list):
                train_time_num = len(train_time_list)
                next = False
            train_time = train_time_list[-train_time_num:]
            train_x = X_train[X_train[primary_time].isin(train_time)]
            train_y = y_train[-train_x.shape[0]:]
            lgb_train = lgb.Dataset(train_x, train_y)
            lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)
            # print('train_time_num: ', train_time_num)
            model = lgb.train(self.params['params'],
                              train_set=lgb_train, valid_sets=lgb_eval, categorical_feature=categorical_feature,
                              valid_names='eval', early_stopping_rounds=30, num_boost_round=10000, verbose_eval=20)
            # print(f'best_score: {model.best_score}')
            sc = model.best_score["eval"][self.params['params']["metric"]]
            if (min_score == -1) or (sc <= min_score):
                min_score = sc
                best_train_time_num = train_time_num
            else:
                break

        return best_train_time_num

    def adjust(self, X_train, y_train, X_eval, y_eval, categorical_feature, round=100):
        params = self.params.copy()
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)

        score = {}
        call_def = lgb.record_evaluation(score)
        self.model = lgb.train(self.params['params'], train_set=lgb_train, valid_sets=[lgb_train, lgb_eval],
                               valid_names=['train', 'eval'], early_stopping_rounds=30, callbacks=[call_def],
                               num_boost_round=round, verbose_eval=10,
                               # learning_rates=0.05
                               )
        best_eval = min(score['eval']['rmse'])
        return self.model.best_score["train"][self.params['params']["metric"]], best_eval, self.score()
