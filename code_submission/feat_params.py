import gc

from default_feat import TimeReversalSimpleDropna

from logger_control import time_limit
from tools import time_train_test_split, data_sample_bykey_rate

history_feat_list = [
    'TimeReversal',
    'TimeReversalSimple',
    'TimeReversalSimple_origin',
    'TimeReversalSimpleDropna',
    'TimeReversalSimpleDropna_origin',
]


class FeatParams:
    def __init__(self, X, tt, interval, natime_num, ):
        self.X = X
        self.df_raw_num = X.data.shape[0]
        self.tt = tt
        self.interval = interval
        self.train_ttnum = X.data[X.primary_timestamp].nunique()
        self.test_ttnum = X.pred_timestamp
        self.natime_num = natime_num
        self.pal_size = 30  # primary_agg label 滑窗大小
        self.tpal_size = 30  # 基于时间primary_agg label 滑窗大小
        self.select_orfeat = []
        self.pao_size = 7  # primary_agg 原始特征 滑窗大小 5*todo_feat_size
        self.pilm_size = 7  # primary_id label 滑窗大小 len(primary_id)*7 primary_id至少两个
        self.tl_size = 7  # 基于时间 label 滑窗大小
        self.palm_size = 3  # primary_agg label滑窗取均值大小 pal_size//palm_size
        self.palt_list = [3, 7]  # primary_agg label滑窗统计信息窗口大小列表 len(palt_list)*4
        self.inl2pal = {'s': 0, 'M': 0, 'h': 0, 'w': 0, 'd': 0, 'm': 0}
        self.dtype_cols = {}
        self.timeReversal_combine = {}
        self.timeReversal_cols = []
        self.drop_feat = []
        self.dropna_flag = True
        self.fillna_flag = True
        self.interval2grid = {'s': [15, 30, 60], 'M': [15, 30, 60], 'h': [12, 24, 48], 'w': [12, 20, 30],
                              'd': [15, 30, 60], 'm': [6, 12, 30], 'y': [3, 5, 10]}
        self.time2inl = {'s': 6, 'M': 6, 'h': 6, 'w': 6, 'd': 7, 'm': 3, 'y': 2}
        self.feat_order1 = []
        self.agg_combine_time = []
        self.agg_combine_num = []

    def agg_combine_time_params(self):
        interval_list = ['second', 'minute', 'hour', 'weekday', 'day', 'month']
        interval_num = [1, 1, 1, 7, 30, 365]
        ss = self.X.data[self.X.primary_timestamp]
        timerange = ss.max() - ss.min()
        all_day = timerange // 86400
        for j, inl in enumerate(interval_list):
            inl_col = self.X.primary_timestamp + '_' + inl
            if inl_col not in self.X.data.columns:
                continue
            if all_day % interval_num[j] == 0:
                max_win = int(all_day/interval_num[j])
            else:
                max_win = int(all_day / interval_num[j])+1
            if max_win > 1:
                self.agg_combine_time.append(inl_col)
                self.agg_combine_num.append(min(max_win-1, 7))

    def fit_transform(self):
        self.initial()
        new_cols = self.origin_cols[:]
        if self.dropna_flag:
            new_cols.extend(self.adjust_label_windom_size_dropna())
        self.select_orfeat = list(set(self.select_orfeat))
        self.adjust_label_reversal()
        self.X = None
        gc.collect()

    def initial(self):
        df_raw_num = self.X.data.shape[0]
        pred_time_num = self.X.data[self.X.primary_timestamp].nunique()
        self.primari_agg_num = self.X.data[self.X.primary_agg].nunique()
        real_raw_num = pred_time_num * self.primari_agg_num
        _rate = df_raw_num / real_raw_num
        if _rate > 0.97:
            self.fillna_flag = False
        if (_rate < 0.1) and (real_raw_num > 8000000):
            self.fillna_flag = False

        self.X.data = data_sample_bykey_rate(self.X.data, self.X.primary_id, self.X.primary_agg, self.X.primary_timestamp)
        self.origin_cols = list(self.X.data.columns)

    def adjust_label_reversal(self):
        max_win_sum = (400000/self.primari_agg_num)
        self.pal_size = int(min(max_win_sum*0.4, 30))
        self.pal_size = max(self.pal_size, 1)
        self.pao_size = int(min(max_win_sum*0.1, min(int(0.30*self.pal_size) + 1, 20)))
        self.pao_size = max(self.pao_size, 1)
        self.pilm_size = max(self.pao_size, 2)
        self.tl_size = max(self.pilm_size, 3)

    def adjust_label_windom_size_dropna(self, search_grid=None):
        if search_grid is None:
            search_grid = [7]

        windom_max = max(search_grid)
        timeReversal = TimeReversalSimpleDropna(self.X.primary_agg, self.X.primary_timestamp, self.X.label, windom_max,
                                                )
        class_name = 'TimeReversalSimpleDropna'+self.X.label
        with time_limit(class_name + '.fit'):
            timeReversal.train_fit(self.X.data)
        with time_limit(class_name + '.transform'):
            new_cols = timeReversal.train_transform(self.X)

        best_score = -1
        orselect_feat = self.X.origin_feat[:]
        for col in self.X.primary_id:
            orselect_feat.remove(col)
        for windom in search_grid:
            train = self.X.data[self.origin_cols+new_cols[:windom]].copy()
            X_ = train
            y = train.pop(self.X.label)
            X_train, y_train, X_eval, y_eval = time_train_test_split(X_, y, self.X.primary_timestamp, shuffle=False)
            X_train.drop(columns=self.X.primary_timestamp, inplace=True)
            X_eval.drop(columns=self.X.primary_timestamp, inplace=True)
            sc_train, sc, imp = self.X.lgb_model.adjust(X_train, y_train, X_eval, y_eval, [], 100)
            if (best_score > sc) or (best_score==-1):
                self.pal_size = windom
                best_score = sc
                self.select_orfeat = list(imp[imp['features'].isin(orselect_feat)]['features'])[:5]

        self.X.data.drop(columns=new_cols[self.pal_size:], inplace=True)
        return new_cols[:self.pal_size]
