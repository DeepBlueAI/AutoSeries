import CONSTANT
from tools import time_train_test_split, data_sample_bykey_rate
import pandas as pd
import lightgbm as lgb


class FeatSelect:
    def __init__(self):
        self.num_boost_round = 100
        num_leaves = 63
        self.params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': "None",
            'learning_rate': 0.1,
            'num_leaves': num_leaves,
            'max_depth': -1,
            'min_child_samples': 20,
            'max_bin': 255,
            'subsample': 0.9,
            'subsample_freq': 1,
            'colsample_bytree': 1,
            'min_child_weight': 0.001,
            'subsample_for_bin': 200000,
            'min_split_gain': 0.02,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'seed': CONSTANT.SEED,
            'nthread': CONSTANT.THREAD_NUM,
         }
        self.len2featn = [(2000000, 75), (1000000, 100), (500000, 120), (100000, 180), (0, 300)]
        self.drop_feat = []

    def feat_select(self, df, keep_cols, primary_timestamp, primary_id, primary_agg, label):
        df_raw_num = df.shape[0]
        X = data_sample_bykey_rate(df, primary_id, primary_agg, primary_timestamp)
        y = X.pop(label)
        X.pop('changed_y')
        X.drop(columns=primary_timestamp, inplace=True)
        imp = self.adjust(X, y, [], 100)
        self.drop_feat = list(imp.loc[imp['importances'] < 1]['features'])
        imp = imp[imp['importances'] > 0]
        imp = list(imp['features'])
        for df_len, feat_num in self.len2featn:
            if df_raw_num > df_len:
                self.drop_feat.extend(imp[feat_num:])
                break
        self.drop_feat = list(set(self.drop_feat) - set(keep_cols))

    def adjust(self, X_train, y_train, categorical_feature, round=100):
        params = self.params.copy()
        params['num_boost_round'] = round
        dtrain = lgb.Dataset(X_train, y_train, feature_name=list(X_train.columns))

        model = lgb.train(params, dtrain,
                          num_boost_round=self.num_boost_round,
                          categorical_feature=categorical_feature,
                          )
        df_imp = pd.DataFrame({'features': [i for i in model.feature_name()],
                               'importances': model.feature_importance()})

        df_imp.sort_values('importances', ascending=False, inplace=True)
        return df_imp
