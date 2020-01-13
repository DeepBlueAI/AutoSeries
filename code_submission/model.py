import copy
import pickle
import time

import pandas as pd
from feat_engine import Feat_engine
from logger_control import time_limit
from models import LGBMRegressor, LinearRegressor
from preprocessing import Preprocess, time_interval, TransExponentialDecay
import os
from feat_params import FeatParams
from tools import time_train_test_split, serch_best_fusion_proportion
import gc


class Model:
    def __init__(self, info, test_timestamp, pred_timestamp):
        self.info = info
        self.primary_timestamp = info['primary_timestamp']
        self.primary_id = info['primary_id']
        self.primary_agg = None
        self.label = info['label']
        self.schema = info['schema']
        self.schema.pop(self.label)
        self.origin_feat = list(self.schema.keys())
        print(f"\ninfo: {self.info}")

        self.dtype_cols = {'cat': [col for col, types in self.schema.items() if types == 'str'],
                           'num': [col for col, types in self.schema.items() if types == 'num']}

        self.test_timestamp = test_timestamp
        self.pred_timestamp = pred_timestamp

        self.n_test_timestamp = len(pred_timestamp)

        self.split_num = 5
        self.update_interval = int(self.n_test_timestamp / self.split_num)

        self.lgb_model = LGBMRegressor()
        self.linear_model = LinearRegressor()

        self.use_Linear = True
        self.use_sample_weight = False
        self.use_exp_y = True

        self.tmpControlType = 4

        self.time_seg = 0

        self.linear_weight = 0
        self.lgb_weight = 0

        self.n_predict = 0
        self.isfirst_predict = True
        self.last_drop_col = []
        self.history = pd.DataFrame()

        self.new_model_n_predict = 0
        self.new_model_history_label = []
        self.lgb_predict_list = []
        self.linear_predict_list = []

        self.train_time_num = 0
        self.preprocess = None
        self.featParamsad = None
        self.feat_engine = None
        self.data = pd.DataFrame()
        self.train_time = 0

    def update_data(self, df):
        self.data = df

    def train(self, train_data, time_info):
        self.new_model_history_label = []
        self.lgb_predict_list = []
        self.linear_predict_list = []
        self.new_model_n_predict = 0

        self.data = train_data
        gc.collect()

        self.data['changed_y'] = self.data[self.label].copy()
        self.preprocess = Preprocess()
        self.preprocess.train_preprocess(self)

        if self.n_predict == 0:
            tt, interval, na_num = time_interval(self.data[self.primary_timestamp])
            with time_limit("featParamsad"):
                self.featParamsad = FeatParams(copy.deepcopy(self), tt, interval, na_num)
                self.featParamsad.fit_transform()

        gc.collect()

        self.feat_engine = Feat_engine(self.featParamsad)
        self.feat_engine.same_feat_train(self)
        self.feat_engine.history_feat_train(self)

        if self.use_sample_weight:
            TransExponentialDecay(self.primary_timestamp, init=1.0, finish=0.75, offset=0).fit(train_data)

        gc.collect()

        col = self.data.any()
        col = col[col].index
        self.data = self.data[col]
        gc.collect()

        X = self.data

        categorical_feature = []
        self.last_drop_col.append(self.primary_timestamp)

        if self.n_predict == 0:
            y = self.data.pop(self.label)
            y1 = self.data['changed_y']
            X_train, y_train, X_eval, y_eval = time_train_test_split(X, y, self.primary_timestamp, shuffle=False)
            if self.time_seg:
                seg_num = len(X_train) // self.time_seg
                X_train['time_seg'] = [(((i // seg_num)+1) if ((i // seg_num)+1) <= self.time_seg else self.time_seg) for i in range(len(X_train))]
                X_eval['time_seg'] = self.time_seg

            self.lgb_model.param_opt_new(X_train, y_train, X_eval, y_eval, categorical_feature, self.primary_id, self.primary_agg, self.primary_timestamp)
            X_train.drop(self.last_drop_col, axis=1, inplace=True)

            _, sc1 = self.lgb_model.valid_fit(X_train, y_train, X_eval, y_eval, categorical_feature,
                                              self.use_sample_weight, round=100)
            if (y != y1).any():
                y_train = y1[:len(y_train)]
                mod1 = self.lgb_model.model
                self.lgb_model.model = None
                _, sc2 = self.lgb_model.valid_fit(X_train, y_train, X_eval, y_eval, categorical_feature,
                                                  self.use_sample_weight, round=100)
                if sc2 < sc1:
                    gc.collect()
                    self.use_exp_y = False
                    y = y1
                else:
                    y_train = y[:len(y_train)]
                    self.lgb_model.model = mod1
            lgb_preds, _ = self.lgb_model.valid_fit(X_train, y_train, X_eval, y_eval, categorical_feature,
                                                    self.use_sample_weight)

            col = X_train.any()
            col = col[col].index
            X_train = X_train[col]
            X_eval = X_eval[col]
            gc.collect()
            linear_preds = self.linear_model.valid_fit(X_train, y_train, X_eval, y_eval, self.use_sample_weight)
            gc.collect()
            if self.tmpControlType == 1:
                self.linear_weight, self.lgb_weight = 1, 0
            elif self.tmpControlType == 2:
                self.linear_weight, self.lgb_weight = 0, 1
            else:
                self.linear_weight, self.lgb_weight = serch_best_fusion_proportion(linear_preds, lgb_preds, y_eval)
        else:
            if not self.use_exp_y:
                self.data[self.label] = self.data['changed_y'].copy()
            y = self.data.pop(self.label)
            self.data.pop('changed_y')

        X.drop(self.last_drop_col, axis=1, inplace=True)

        if self.time_seg:
            seg_num = len(X) // self.time_seg
            X['time_seg'] = [(((i // seg_num) + 1) if ((i // seg_num) + 1) <= self.time_seg else self.time_seg) for i in range(len(X))]

        with time_limit("linear_fit"):
            self.linear_model.fit(X, y, self.use_sample_weight)

        with time_limit("fit"):
            self.lgb_model.fit(X, y, categorical_feature, self.use_sample_weight)
        next_step = 'predict'
        return next_step

    def after_train(self):
        pass

    def predict(self, new_history, pred_record, time_info):
        if (time_info['predict'] < 5) and not new_history.empty:
            if self.primary_id:
                lab_list = pred_record.join(new_history.set_index(self.primary_id)[self.label], how='left', on=self.primary_id)
                lab_list = lab_list[self.label].fillna(new_history[self.label].mean())
            else:
                lab_list = pred_record.shape[0]*list(new_history[self.label])[-1:]
            return list(lab_list), 'predict'
        self.data = pred_record

        if not new_history.empty:
            y = new_history[self.label]
            self.history[self.label] = y
            if len(self.linear_predict_list):
                self.new_model_history_label.extend(list(new_history[self.label]))

        if self.tmpControlType == 4:
            if ((self.new_model_n_predict >= 50) and ((self.new_model_n_predict % 50) == 0)) or (self.new_model_n_predict == 15):
                linear_weight, lgb_weight = serch_best_fusion_proportion(pd.Series(self.linear_predict_list),
                                                                                   pd.Series(self.lgb_predict_list), pd.Series(self.new_model_history_label))
                self.linear_weight = self.linear_weight*0.5 + linear_weight*0.5
                self.lgb_weight = self.lgb_weight*0.5 + lgb_weight*0.5
            self.new_model_n_predict += 1

        # preprocess
        self.preprocess.test_preprocess(self)

        # feat_engine
        self.feat_engine.same_feat_test(self)
        hh = self.data.copy()
        self.feat_engine.history_feat_test(self)
        self.history = hh
        self.n_predict += 1

        self.data.drop(self.last_drop_col, axis=1, inplace=True)

        if self.time_seg:
            self.data['time_seg'] = self.time_seg

        linear_preds = self.linear_model.predict(self.data)
        lgb_preds = self.lgb_model.predict(self.data)
        predictions = self.linear_weight*linear_preds + self.lgb_weight*lgb_preds
        self.lgb_predict_list.extend(list(lgb_preds))
        self.linear_predict_list.extend(list(linear_preds))

        if (self.n_predict % self.update_interval == 0) and (self.n_predict < self.split_num*self.update_interval) and (time_info['update'] > self.train_time*1.25):
            next_step = 'update'
            self.feat_engine = None
            self.preprocess = None
            self.history = pd.DataFrame()
            self.isfirst_predict = True
            self.new_model_history_label = None
            self.lgb_predict_list = None
            self.linear_predict_list = None
            gc.collect()
        else:
            self.isfirst_predict = False
            next_step = 'predict'
        if self.n_predict == self.n_test_timestamp:
            self.feat_engine = None
            self.preprocess = None
            self.history = pd.DataFrame()
            self.isfirst_predict = True
            self.new_model_history_label = None
            self.lgb_predict_list = None
            self.linear_predict_list = None
            gc.collect()
        return list(predictions), next_step

    def update(self, train_data, test_history_data, time_info):
        t1 = time.time()
        print(f"\nUpdate time budget: {time_info['update']}s")

        total_data = pd.concat([train_data, test_history_data])

        total_data.drop_duplicates(subset=[self.primary_timestamp]+self.primary_id, inplace=True)
        total_data.reset_index(drop=True, inplace=True)
        self.train(total_data, time_info)

        print("Finish update\n")
        self.train_time = time.time()-t1
        next_step = 'predict'
        return next_step

    def save(self, model_dir, time_info):
        print(f"\nSave time budget: {time_info['save']}s")
        self.data = pd.DataFrame()
        gc.collect()
        pkl_list = []

        for attr in dir(self):
            if attr.startswith('__') or attr in ['train', 'predict', 'update', 'save', 'load']:
                continue

            pkl_list.append(attr)
            pickle.dump(getattr(self, attr), open(os.path.join(model_dir, f'{attr}.pkl'), 'wb'))

        pickle.dump(pkl_list, open(os.path.join(model_dir, f'pkl_list.pkl'), 'wb'))

        print("Finish save\n")

    def load(self, model_dir, time_info):
        print(f"\nLoad time budget: {time_info['load']}s")

        pkl_list = pickle.load(open(os.path.join(model_dir, 'pkl_list.pkl'), 'rb'))

        for attr in pkl_list:
            setattr(self, attr, pickle.load(open(os.path.join(model_dir, f'{attr}.pkl'), 'rb')))

        print("Finish load\n")
