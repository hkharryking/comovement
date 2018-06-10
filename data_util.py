from sklearn import preprocessing
import pandas as pd
import numpy as np
import itertools

class Data_util:

    def __init__(self,ticker_num, window, feature_num, input_path,target_path):
        self.dict_dyadic = {}  # inverted dict to store the [i,j] to the h-th enumeration result
        self.dict_ticker = {}  # dict to store the ticker name of the i-th ticker in the raw data
        self.ticker_num = ticker_num
        self.input_path = input_path
        self.target_path = target_path
        self.window = window
        self.feature_num = feature_num
        self.compare_data = None
        self.security_table = pd.read_csv('/data/securities.csv')

    def get_dyadic_size(self):
        if self.compare_data is not None:
            return self.compare_data.shape[0]
        return None

    def check_dyadic(self, i):
        return self.dict_dyadic[i]

    def check_ticker(self, i):
        return self.dict_ticker[i]

    def check_index(self,ticker_name):
        for i in range(0,len(self.dict_ticker)):
            if self.dict_ticker[i]==ticker_name:
                return i
        return -1

    def check_sector(self,ticker_name):
        for security in self.security_table.iterrows():
            if security[1]['Ticker symbol'] == ticker_name:
                return security[1]['GICS Sector']
        return None


    def read_data(self, fname):
        """
        read the raw csv data as a pandas dataframe
        """
        df = pd.read_csv(fname, index_col=0, parse_dates=True)
        df["adj close"] = df.close  # Moving close to the last column as the predcit label
        df.drop(['open'], 1, inplace=True)
        df.drop(['close'], 1, inplace=True)  # Moving close to the last column
        df.drop(['high'], 1, inplace=True)
        df.drop(['low'], 1, inplace=True)
        #df.drop(['volume'], 1, inplace=True)
        return df

    def normalize_data(self, df):
        """
        nomalize the data with the min max scaler to each feature
        """
        min_max_scaler = preprocessing.MinMaxScaler()
        # df['open'] = min_max_scaler.fit_transform(df.open.values.reshape(-1,1))
        #df['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1, 1))
        #df['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1, 1))
        df['volume'] = min_max_scaler.fit_transform(df.volume.values.reshape(-1, 1))
        df['adj close'] = min_max_scaler.fit_transform(df['adj close'].values.reshape(-1, 1))
        return df

    def get_full_index(self, groups):
        maxindexnumber = groups.count().max()['adj close']
        for g in groups:
            if g[1].count()['adj close'] == maxindexnumber:  # tuple:(symbol, DataFrame)
                return g[1].index  # tuple:(symbol, DataFrame)
        return None

    def get_groups(self):
        if self.groups != None:
            #len(tickers), actual_time_steps, self.feature_num)
            data = self.groups # warning: have to evoke after load_x
            data = data[data.shape[0],data.shape[1],0] # high
            outputs = []
            all_edge_combinations = [c for c in itertools.combinations(range(self.ticker_num),2)]
            for (i,j) in all_edge_combinations:
                outputs.append(np.concatenate(data[(i,j),:,:],axis=1))

    def group_select(self, df):
        """
        group the data by the tickers (symbol)
        """
        tickers = list(set(df.symbol))
        tickers.sort() # sort according to the tickers here
        #df=df.sort_values(by='symbol')
        tickers = tickers[0:self.ticker_num]
        df = df[df.symbol.isin(tickers)]
        actual_time_steps = len(set(df.index))
        remander = actual_time_steps % self.window
        if remander != 0:
            # to keep the length of time series is the multiple of TIME_STEP
            actual_time_steps = int(actual_time_steps / self.window) * self.window
        result = []
        # df.sort_values('symbol', ascending = False, inplace=True)
        groups = df.groupby('symbol',sort=True)#.apply(lambda x: x.order)
        fullindex = self.get_full_index(groups)
        i = 0
        for group in groups:
            # group=tuple:(symbol, DataFrame)
            ticker_name = group[0]  # tuple:(symbol, DataFrame)
            #print(ticker_name)
            group = group[1]  # tuple:(symbol, DataFrame)
            group.sort_index()
            self.dict_ticker[i] = ticker_name

            i += 1

            if len(group.index) < len(fullindex):  # align all the time series
                group = group.reindex(fullindex).fillna(0)
            length = 0
            for item in group.drop('symbol', 1).as_matrix():  # tuple:(symbol, DataFrame)
                result.append(item)
                length += 1
                if length >= actual_time_steps:
                    break

        result = np.array(result)
        return result.reshape(len(tickers),actual_time_steps,self.feature_num)

    def timeseries_enumerate(self, data):
        """
        enumerate all the combinations of the timeseries from data
        data (timeseries_num, time_steps, feature_number)
        return following format nn.Conv2d input format：batch x channel(time steps for time series) x height x width
        e.g. t11,t12,t13,t14....
             t21,t22,t23,t24....
             ...
        """
        outputs = []       # for my methods
        out_comparison=[]  # for compare methods
        all_edge_combinations = [c for c in itertools.combinations(range(self.ticker_num), 2)]
        c = 0
        for (i, j) in all_edge_combinations:
            self.dict_dyadic[c] = (i, j)
            c += 1
            outputs.append(np.concatenate(data[(i, j), :, :], axis=1).transpose())
            #out_comparison.append(data[(i, j), :, 3]) # close
            out_comparison.append(data[(i, j), :, 0])  # index for only 1 dimension data close
        outputs = np.array(outputs)
        out_comparison = np.array(out_comparison)
        self.compare_data = out_comparison.reshape(len(all_edge_combinations),2, out_comparison.shape[2]) #format：Samples , 2, width (time steps)
        #outputs=outputs.reshape(len(all_edge_combinations), self.feature_num * 2, outputs.shape[1])
        return outputs
        # nn.Conv1d input format：Samples , channel(features) , width (time stamps)


    def load_x(self, period):
        print("Loaded input data...")
        data = self.read_data(self.input_path)
        start, end = period
        data = data[start:end]
        data = self.normalize_data(data)
        return self.timeseries_enumerate(self.group_select(data))

    def get_concatenated_time_series(self,ticker_dyadics,period):
        outputs = []
        print("Loaded input data...")
        data = self.read_data(self.input_path)
        start, end = period
        data = data[start:end]
        data = self.normalize_data(data)
        data = self.group_select(data)
        for ticker1,ticker2 in ticker_dyadics:
            i = self.check_index(ticker1)
            j = self.check_index(ticker2)
            outputs.append(np.concatenate(data[(i, j), :, :], axis=1).transpose())
        outputs = np.array(outputs)
        return outputs


    def load_y(self, period):
        print("Loaded target data...")
        data = self.read_data(self.target_path)
        start, end = period
        daily_pct_change = data[start:end]['adj close'].groupby('date').sum().pct_change()
        ys = daily_pct_change.fillna(0).as_matrix()
        # vol = daily_pct_change.rolling(MIN_PERIODS).std() * np.sqrt(MIN_PERIODS)
        # ys = vol.fillna(0).as_matrix()
        ys = np.array([1 if i > 0 else 0 for i in ys])

        remainder = int(len(ys)) % self.window
        if remainder != 0:
            # keep the time series to the multiple of the TIME_STEPS
            ys = ys[0:int(int(len(ys) / self.window) * self.window)]
        ys = ys[self.window:]  # step of the first window
        return ys

    def load_target(self, period):
        print("Loaded target data...")
        data = self.read_data(self.target_path)
        start, end = period
        data = data[start:end]['adj close'].groupby('date').sum()
        return data.as_matrix()
