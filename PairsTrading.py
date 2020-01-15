# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 19:01:47 2019

@author: Xinyu Xi, Jierui Wang, Yihan He, Siyi Huang, Guoliang Sheng
"""
import pandas as pd
import statsmodels.api as sm
import numpy as np
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from pykalman import KalmanFilter


class PairsTrading:
    def __init__(self, path, method, start, trade_start, end, price_t, error_t):
        self.data = self.get_data(path, start, end)
        self.method = method
        self.holding = False
        self.log_data = self.get_previous_standard_data(trade_start, end)
        self.price_t = price_t
        self.error_t = error_t

    @staticmethod
    # get original price data
    def get_data(path, start, end):
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index)
        df = df[start:end]
        return df

    # author: Xinyu Xi
    def find_cointegrated_pairs(self):
        # 选最初一年的log(price)
        data_find = np.log(self.data[:str(self.data.index[0].year+1)])
        # 得到DataFrame长度
        n = data_find.shape[1]
        # 初始化t-statistic值矩阵
        t_value = pd.DataFrame()
        # 抽取列的名称
        keys = data_find.keys()
        # 初始化强协整组
        pairs = []
        # 对于每一个i
        for i in range(n):
            # 对于大于i的j
            for j in range(i + 1, n):
                # 获取相应的两只股票的价格Series
                stock1 = data_find[keys[i]]
                stock2 = data_find[keys[j]]
                # 分析它们的协整关系
                result = sm.tsa.stattools.coint(stock1, stock2)
                # 取出并记录t值
                tvalue = result[0]
                # 如果t值小于-3
                if tvalue < self.price_t:
                    # 记录股票对和相应的t值
                    pairs.append([keys[i], keys[j]])
                    t_value.loc[keys[i] + '&' + keys[j], "t_value"] = tvalue
        return t_value, pairs

    # rolling every month, get one-year data
    # author: Siyi Huang, Xinyu Xi
    def get_previous_standard_data(self, trade_start, end):
        log_ret = []
        for i in pd.date_range(trade_start, end, freq='bm'):
            year = i.year
            month = i.month
            # Get previous one year's data
            if month == 12:
                df_i = self.data[str(year) + '-01':str(year) + '-12']
            else:
                df_i = self.data[str(year - 1) + '-' + str(month + 1):
                                 str(year) + '-' + str(month)]
            log_ret.append(np.log(df_i) - np.log(df_i.iloc[0]))
        return log_ret

    # get trade data
    # author: Siyi Huang, Xinyu Xi
    def get_monthly_standard_data(self):
        data_trade = self.data[str(self.data.index[0].year+1):]
        monthly_date = pd.date_range(start=data_trade.index[0],
                                     end=data_trade.index[-1], freq='m')
        log_ret = pd.DataFrame()
        for i in monthly_date:
            data = data_trade[str(i.year) + "-" + str(i.month)]
            data = np.log(data) - np.log(data.iloc[0])

            log_ret = log_ret.append(data)
        return log_ret

    @staticmethod
    # author: Jierui Wang
    def OLS(daily, pair):
        Y = daily[pair[0]]
        X = sm.add_constant(daily[pair[1]])
        model = sm.OLS(Y, X, hasconst=True)
        res = model.fit()
        alpha = res.params[0]  # first one in params is the constant
        beta = res.params[1]  # and the second is beta
        error = Y - np.dot(res.params.T, X.T)  # append each time series
        return alpha, beta, error

    @staticmethod
    # author: Jierui Wang
    def TLS(daily, pair):
        # Here we use the numerical solution
        x_bar = daily[pair[1]].mean()
        y_bar = daily[pair[0]].mean()
        c0 = ((daily[pair[1]] - x_bar) * (daily[pair[0]] - y_bar)).sum()
        c1 = ((daily[pair[1]] - x_bar) ** 2 -
              (daily[pair[0]] - y_bar) ** 2).sum()
        c2 = - c0
        # !!!Should we +/- the sqrt one?????? Answer:+
        beta1 = (-c1 + np.sqrt(c1 ** 2 - 4 * c0 * c2)) / (2 * c0)
        beta0 = y_bar - beta1 * x_bar
        # ibsl = y - kx - a
        error = daily[pair[0]] - beta1 * daily[pair[1]] - beta0
        return beta0, beta1, error

    # author: Yihan He
    def calculate_kalman_filter1(self, pair):
        y = pd.DataFrame(self.data[pair[0]])
        x = pd.DataFrame(self.data[pair[1]])
        daily1 = self.data[self.data.index.year > self.data.index.year[0]]
        daily_1st = self.data[str(self.data.index.year[0])]
        year_list = set(daily1.index.year)
        year_list1 = set(daily_1st.index.year)
        daily_ = []
        for i in year_list:
            daily_i = np.log(daily1[str(i)]) - np.log(daily1[str(daily1.index.year[0])].iloc[0])
            daily_.append(daily_i)
        for j in year_list1:
            daily_1st_j = np.log(daily_1st[str(j)]) - np.log(daily_1st[str(daily_1st.index.year[0])].iloc[0])
        daily = daily_1st_j.append(daily_)
        # 从2002年开始滚动
        start_index = np.where((daily.index).year == daily1.index.year[0])[0][0]
        error = []
        std = []
        adf = []
        observation_matrices = np.vstack((np.ones(len(daily_1st_j)), daily_1st_j[pair].iloc[:, 1].values)).T
        shape = observation_matrices.shape
        observation_matrices = observation_matrices.reshape(shape[0], 1, shape[1])
        kf = KalmanFilter(transition_matrices=np.array([[1, 0], [0, 1]]),
                          observation_matrices=observation_matrices)  # 转移矩阵为单位矩阵#
        np.random.seed(0)  # 使用第一年的数据，采用Em算法，估计出初始状态
        kf.em(daily_1st_j[pair].iloc[:, 0])  # 初始状态的协方差，观测方程和状态方程误差的协方差
        filter_mean, filter_cov = kf.filter(daily_1st_j[pair].iloc[:, 0])  # 对数据做滤波

        for i in range(start_index, len(daily)):

            observation = y.values[i]
            observation_matrices = np.array([[1, daily[pair[1]].values[i]]])
            next_filter_mean, next_filter_cov = kf.filter_update(filtered_state_mean=filter_mean[-1],
                                                                 filtered_state_covariance=filter_cov[-1],
                                                                 observation=observation,
                                                                 observation_matrix=observation_matrices)
            filter_mean1 = np.vstack((filter_mean, next_filter_mean))
            # filter_cov1 = np.vstack((filter_cov,next_filter_cov.reshape( 1, 2, 2)))
            alpha = pd.Series(filter_mean1[start_index - 1:, 0])  # 得到alpha和beta
            beta = pd.Series(filter_mean1[start_index - 1:, 1])
            error0 = daily[pair].iloc[i, 0] - daily[pair].iloc[i, 1] * filter_mean1[i - 1, 1] - filter_mean1[i - 1, 0]
            error.append(error0)
            days_num = daily[str(daily.index[i].year)].shape[0]
            error_t = []
            observation_matrix = np.vstack((np.ones(len(daily[pair].iloc[0:i, :])), x.iloc[:i, 0].values)).T
            shape = observation_matrix.shape
            observation_matrix = observation_matrix.reshape(shape[0], 1, shape[1])  # 定义卡尔曼滤波的方程
            kf = KalmanFilter(transition_matrices=np.array([[1, 0], [0, 1]]),
                              observation_matrices=observation_matrix)  # 转移矩阵为单位矩阵#
            np.random.seed(0)  # 使用第一年的数据，采用Em算法，估计出初始状态
            kf.em(y[:(i)])  # 初始状态的协方差，观测方程和状态方程误差的协方差
            filter_mean, filter_cov = kf.filter(y[:(i)])  # 对数据做滤波
            for j in range((i - days_num), i):
                error_test = daily[pair].iloc[j, 0] - daily[pair].iloc[j, 1] * filter_mean1[i - 1, 1] - filter_mean1[
                    i - 1, 0]

                error_t.append(error_test)
            adf_test_t = adfuller(error_t, regression='ct', autolag="BIC")[0]
            adf.append(adf_test_t)
            std_t = pd.Series(error_t).std()
            std.append(std_t)

        error = pd.DataFrame(error)
        adf = pd.DataFrame(adf)
        std = pd.DataFrame(std)
        adf.columns = ["t"]
        adf.index = daily1.index
        alpha.columns = ["alpha"]
        std.columns = ["beta"]
        alpha.index = daily1.index
        beta.index = daily1.index
        error.index = daily1.index
        beta.columns = ["beta"]
        error.columns = ["error"]
        std.index = daily1.index
        daily_param = pd.concat([error, alpha, beta,  std, adf], axis=1)
        daily_param.columns = ["error", "alpha", "beta", "std", "t"]
        return daily_param

    # author: Xinyu Xi
    def get_trading_params(self, pair):
        # monthly data: alpha, beta, std, tvalues from last 1 year
        params = pd.DataFrame()
        for i in range(0, len(self.log_data)):
            if self.method == "OLS":
                alpha, beta, error = self.OLS(self.log_data[i], pair)
            elif self.method == "TLS":
                alpha, beta, error = self.TLS(self.log_data[i], pair)
            elif self.method == "KF":
                pass
            adf_test_t = adfuller(error, regression='ct', autolag='BIC')[0]
            if self.log_data[i].index[-1].month == 12:
                today = str(self.log_data[i].index[-1].year + 1) + "-1"
            else:
                today = str(self.log_data[i].index[-1].year) + "-" + \
                        str(self.log_data[i].index[-1].month + 1)

            params.loc[today, "alpha"] = alpha
            params.loc[today, "beta"] = beta
            params.loc[today, "std"] = error.std()
            params.loc[today, "t"] = adf_test_t

        # convert to daily data: calculate error with actual price
        daily_param = pd.DataFrame(columns=["error", "alpha", "beta", "std", "t"])
        for i in params.index:
            daily_param_i = pd.DataFrame()
            data_i = np.log(self.data[i]) - np.log(self.data[i].iloc[0])
            daily_param_i["error"] = data_i[pair[0]] - params.loc[i, "beta"] * \
                                     data_i[pair[1]] - params.loc[i, "alpha"]
            daily_param_i["alpha"] = params.loc[i, "alpha"] * np.ones(
                daily_param_i.shape[0])
            daily_param_i["beta"] = params.loc[i, "beta"] * np.ones(
                daily_param_i.shape[0])
            daily_param_i["std"] = params.loc[i, "std"] * np.ones(
                daily_param_i.shape[0])
            daily_param_i["t"] = params.loc[i, "t"] * np.ones(
                daily_param_i.shape[0])
            if i == params.index[0]:
                daily_param = daily_param_i
            else:
                daily_param = daily_param.append(daily_param_i)
        return daily_param

    # author: Jierui Wang, Xinyu Xi
    def trade_nokeep(self, pairs):
        All_Rs = pd.DataFrame()
        pair_adf = []
        t_adf = []
        for pair in pairs:
            if self.method == "KF":
                param = self.calculate_kalman_filter(pair)
            else:
                t, ch, param = self.get_trading_params(pair)
            while ch:
                t_adf.append(t)
                pair_adf.append(pair)
                print("This is pair:", pair)

                #################################################################
                R = pd.Series(index=param.index)
                for i in range(0, len(param.index)):
                    # zero position
                    error = param.iloc[i, 0]

                    while self.holding:
                        error_1 = param.iloc[i - 1, 0]
                        if error * error_1 <= 0:
                            print("**********sell, today is ", param.index[i])
                            self.holding = False
                            R.iloc[i] = 2 * std
                        if i == len(param.index) - 1:
                            print("**********QXPC, today is ", param.index[i])
                            R.iloc[i] = 2 * std - abs(error)
                        break

                    while not self.holding:
                        std = param.iloc[i, 3]
                        if abs(error) >= 2 * std:
                            print("**********buy, today is ", param.index[i])
                            self.holding = True
                        break

                plt.plot(param["error"], label=pair)
                All_Rs[pair[0] + '&' + pair[1]] = R
                break
        plt.legend()
        plt.show()
        return t_adf, All_Rs

    # author: Jierui Wang, Xinyu Xi
    def trade(self, pairs):
        data_trade = self.get_monthly_standard_data()
        All_Rs = pd.DataFrame()
        All_es = pd.DataFrame()

        for pair in pairs:
            print("This is pair:", pair)
            self.holding = False
            param = self.get_trading_params(pair)
            Y = data_trade[pair[0]]
            X = data_trade[pair[1]]

            R = pd.Series(index=param.index)
            e = pd.Series(index=param.index)
            alpha = param.iloc[0, 1]
            beta = param.iloc[0, 2]
            for i in range(0, len(param.index)):
                t = param.iloc[i, 4]
                if t < self.error_t:

                    #################################################################
                    # zero position
                    error = Y.iloc[i] - beta * X.iloc[i] - alpha
                    e.iloc[i] = error

                    while self.holding:
                        error_1 = Y.iloc[i - 1] - beta * X.iloc[i - 1] - alpha
                        if error * error_1 <= 0:
                            print("**********SELL, today is ", param.index[i])
                            self.holding = False
                            R.iloc[i] = 2 * std
                        if i == len(param.index) - 1:
                            print("**********CLOSE POSITION, today is ", param.index[i])
                            R.iloc[i] = 2 * std - abs(error)
                        break

                    while not self.holding:
                        std = param.iloc[i, 3]
                        if abs(error) >= 2 * std:
                            print("**********BUY, today is ", param.index[i])
                            self.holding = True
                            alpha = param.iloc[i, 1]
                            beta = param.iloc[i, 2]
                        break

                    All_Rs[pair[0] + '&' + pair[1]] = R
                    All_es[pair[0] + '&' + pair[1]] = e

                    #################################################################
                else:
                    while self.holding:
                        print("**********CLOSE POSITION, today is ", param.index[i])
                        R.iloc[i] = 2 * std - abs(error)
                        self.holding = False
                        break

        return All_Rs, All_es


if __name__ == "__main__":
    pt = PairsTrading("data.csv", "TLS", "2015-1", "2015-12", "2019-9", -3.5, -2)
    t_price, pairs_coi = pt.find_cointegrated_pairs()
    R, error = pt.trade(pairs_coi)
    R.cumsum(axis=1).mean()
    r = R.fillna(0)
    PandL = r.cumsum().mean(axis=1)
    plt.plot(PandL, label="P/L")
    plt.legend()
    plt.show()
