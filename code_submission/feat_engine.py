from default_feat import TimeReversalSimpleDropna, TimeReversalMeanSimpleDropna, \
    TimeReversalMeanSimpleNoKey, TimeReversalSimple, PrimaryAggFast
from feat_select import FeatSelect
from logger_control import time_limit

feat_history = [
    'TimeReversalSimple',
    'TimeReversalSimpleDropna',
    'TimeReversalSimpleDropna_origin',
    'TimeReversalMeanSimpleDropna',
    'TimeReversalMeanSimpleDropnaNoKey',
]


class Feat_engine:
    def __init__(self, feat_params):
        self.isfirst = True
        self.feat_params = feat_params
        self.same_feat = []
        self.gener_feat = []
        self.timeReversals = []
        self.featSelect = FeatSelect()
        self.time_agg_names = []
        self.agg_max_wins = []

    def same_feat_train(self, X):
        name_list = []

        if 'TimeReversalCombineTime' in feat_history:
            for inl_col in self.feat_params.agg_combine_time:
                feat_cl = PrimaryAggFast([X.primary_agg, inl_col])
                self.same_feat.append(feat_cl)
                name_list.append('KeyLabelBin')
                self.time_agg_names.append(feat_cl.get_feat_name())
        for i, feat_cl in enumerate(self.same_feat):
            with time_limit(name_list[i]):
                feat_cl.fit_transform(X, 'train')

    def same_feat_test(self, X):

        for feat_cl in self.same_feat:
            feat_cl.fit_transform(X, 'test')

    def history_feat_train(self, X):
        name_list = []
        if self.feat_params.dropna_flag:
            if 'TimeReversalSimpleDropna' in feat_history:
                grid = self.feat_params.pal_size
                timeReversal = TimeReversalSimpleDropna(X.primary_agg, X.primary_timestamp, X.label, grid,
                                                        self.feat_params.palm_size,
                                                        self.feat_params.palt_list,
                                                        feat_exp=True
                                                        )
                self.timeReversals.append(timeReversal)
                name_list.append(f'TimeReversalSimpleDropna__{X.label} ({1}, {grid})')

            if 'TimeReversalSimpleDropna_origin' in feat_history:
                grid = self.feat_params.pao_size
                for col in self.feat_params.select_orfeat:
                    timeReversal = TimeReversalSimpleDropna(X.primary_agg, X.primary_timestamp, col, grid)
                    self.timeReversals.append(timeReversal)
                    name_list.append(f'TimeReversalSimpleDropna__{col} ({1}, {grid})')

            if 'TimeReversalCombineTime' in feat_history:
                for win,time_agg_name in zip(self.feat_params.agg_combine_num, self.time_agg_names):
                    timeReversal = TimeReversalSimpleDropna(time_agg_name, X.primary_timestamp, X.label, win,
                                                            )
                    self.timeReversals.append(timeReversal)
                    name_list.append(f'TimeReversalSimpleDropna__{time_agg_name} ({1}, {win})')

        if 'TimeReversalMeanSimpleDropna' in feat_history:
            if len(X.primary_id) > 1:
                grid = self.feat_params.pao_size
                for col in X.primary_id:
                    timeReversal = TimeReversalMeanSimpleDropna(col, X.primary_timestamp, X.label, grid)
                    self.timeReversals.append(timeReversal)
                    name_list.append(f'TimeReversalMeanSimpleDropna__{col} ({1}, {grid})')

        if 'TimeReversalMeanSimpleDropnaNoKey' in feat_history:
            grid = self.feat_params.tl_size
            timeReversal = TimeReversalMeanSimpleNoKey(X.primary_timestamp, X.label, grid)
            self.timeReversals.append(timeReversal)
            name_list.append(f'TimeReversalMeanSimpleDropnaNoKey__{X.label} ({1}, {grid})')

        for i, timeReversal in enumerate(self.timeReversals):
            with time_limit(name_list[i]+'fit'):
                timeReversal.train_fit(X.data)
            with time_limit(name_list[i] + '.transform'):
                timeReversal.train_transform(X)

        if self.feat_params.fillna_flag:
            with time_limit('TimeReversalSimple init'):
                df = X.data[[X.primary_agg, X.primary_timestamp, X.label]]
                df = df.pivot(index=X.primary_agg
                              , columns=X.primary_timestamp, values=X.label).stack(
                    dropna=False).reset_index()
                df.columns = [X.primary_agg, X.primary_timestamp, X.label]
                df = df.sort_values(by=[X.primary_agg, X.primary_timestamp])
                df_index = df.index[df[X.label].notna()]
                df[X.label] = df.groupby(X.primary_agg)[X.label].bfill()

            if 'TimeReversalSimple' in feat_history:
                grid = self.feat_params.pal_size
                timeReversal = TimeReversalSimple(X.primary_agg, X.primary_timestamp, X.label, grid,
                                                  self.feat_params.palm_size,
                                                  self.feat_params.palt_list,
                                                  feat_exp=True
                                                  )
                self.timeReversals.append(timeReversal)
                with time_limit(f'TimeReversalSimple__{X.label} ({1}, {grid})'):
                    timeReversal.train_transform(df)
                name_list.append(f'TimeReversalSimple__{X.label} ({1}, {grid})')

            if 'TimeReversalSimple_origin' in feat_history:
                grid = self.feat_params.pao_size
                df = df.join(X.data.set_index([X.primary_agg, X.primary_timestamp])[self.feat_params.select_orfeat],
                             how='left', on=[X.primary_agg, X.primary_timestamp])
                for col in self.feat_params.select_orfeat:
                    df[X.label] = df.groupby(X.primary_agg)[col].bfill()
                    timeReversal = TimeReversalSimple(X.primary_agg, X.primary_timestamp, col, grid)
                    self.timeReversals.append(timeReversal)
                    with time_limit(f'TimeReversalSimple__{col} ({1}, {grid})'):
                        timeReversal.train_transform(df)
                    name_list.append(f'TimeReversalSimple__{col} ({1}, {grid})')
                df.drop(columns=self.feat_params.select_orfeat, inplace=True)
            df = df.loc[df_index].set_index([X.primary_agg, X.primary_timestamp])
            df.drop(columns=X.label, inplace=True)
            X.data = X.data.join(df, how='left', on=[X.primary_agg, X.primary_timestamp])

        keep_cols = X.primary_id+[X.primary_timestamp, X.primary_agg]
        self.featSelect.feat_select(X.data, keep_cols, X.primary_timestamp, X.primary_id, X.primary_agg, X.label)
        X.data.drop(columns=self.featSelect.drop_feat, inplace=True)

    def history_feat_test(self, X):
        for i, timeReversal in enumerate(self.timeReversals):
            if not self.isfirst:
                timeReversal.test_fit(X.history)

            if timeReversal.__class__.__name__ in ['TimeReversalSimpleDropna', 'TimeReversalSimple']:
                timeReversal.test_transform(X, set(self.featSelect.drop_feat))
            else:
                timeReversal.test_transform(X)

        self.isfirst = False
        drop_feat = list(set(X.data.columns) & set(self.featSelect.drop_feat))
        X.data.drop(columns=drop_feat, inplace=True)

