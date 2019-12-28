#%%
import pandas as pd
import numpy as np
import sys
from itertools import groupby
import math
import pandas_datareader as pdr
from datetime import date, timedelta
import plotly.plotly as py
from plotly import offline
from plotly import graph_objs as go
from plotly.tools import make_subplots
import warnings
pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)

#%%
# def volume_max_signal(stock_id, stock_name, data):
#     if (data.tail(1)['MA_5'][0] > data.tail(1)['MA_20'][0]):
#         if (max(data.tail(10)['Volume']) == data.tail(1)['Volume'][0]):
#             if data.tail(1)['Close'][0] >= data.iloc[-2, data.columns.get_loc('Close')]:
#                 print(stock_id + ' ' + stock_name + ' 大單訊號')
#                 print(stock_id + ' ' + stock_name + ' 大單訊號', file=open("./result/Result.txt", "a"))

# def ma5_signal(stock_id, stock_name, data):
#     if (data.tail(1)['MA_5'][0] > data.tail(1)['MA_20'][0]):
#         if (data.tail(2)['MA_5'][0] < data.tail(2)['MA_20'][0]):
#             print(stock_id + ' ' + stock_name + ' 短日線訊號')
#             print(stock_id + ' ' + stock_name + ' 短日線訊號', file=open("./result/Result.txt", "a"))

#%%
# if __name__ == '__main__':
#     print('System Started')
#     print('__________', file=open("./result/Result.txt", "w"))
#     startTime = date.today() - timedelta(days=50)
#     endTime = date.today()
#     stock_list = pd.read_csv('./data/list.csv', encoding="cp950")
#     for i in range(len(stock_list)):
#         stock_id = str(stock_list['Id'].iloc[i])
#         stock_name = stock_list['Name'].iloc[i]
#         df = pdr.DataReader(stock_id + '.TW', 'yahoo', startTime, endTime)
#         df['MA_5'] = df['Close'].rolling(5).mean()
#         df['MA_20'] = df['Close'].rolling(20).mean()
#         volume_max_signal(stock_id, stock_name, df)
#         ma5_signal(stock_id, stock_name, df)
#         df.to_csv('./data/' + stock_id + stock_name + '.csv')
    
#     print('System Ended')

#%%
class stock_trader():
    def __init__(self):
        self.stock_id = 0
        self.stock_name = np.nan
        self.data = np.nan
        self.ma = False
        self.ma_period = [5, 10, 20]
        self.sar = False
        self.bbands = False
        self.bbands_nk = [20, 2.1]
        self.kd = False
        self.macd = False
        self.account_level = 1000000
        
        # Backtest
        self.do_long = False
        self.do_sell = False
        self.current_level = [self.account_level, self.account_level]
        self.long_price = 0
        self.quantity_on_hand = 0
        self.tp_stop_gain = 0
        self.tp_stop_loss = 0

        # Trade
        self.stock_on_hand = 0
        self.company_list = 0

        return

    def set_backtest_stock_id_name(self, stock_id, stock_name):
        self.stock_id = stock_id
        self.stock_name = stock_name

    def get_data(self, stock_id):
        if stock_id == 0:
            return False
        startTime = date.today() - timedelta(days=400)
        endTime = date.today()
        self.data = pdr.DataReader(stock_id + '.TW', 'yahoo', startTime, endTime)
        
        return True

    def set_moving_average_period(self, p1, p2, p3):
        self.ma_period = [p1, p2, p3]

        return

    def add_moving_average(self):
        self.ma = True
        for period in self.ma_period:
            name = 'MA_' + str(period)
            if name not in self.data.columns:
                self.data[name] = self.data.Close.rolling(period).mean()
                # print('{} added' .format(name))
        
        return 

    def add_parabolic_sar(self):
        self.sar = True
        # print('SAR added')

        iaf = 0.02
        maxaf = 0.2
        high = list(self.data.High)
        low = list(self.data.Low)
        close = list(self.data.Close)
        length = len(self.data)
        psar = close[0:len(close)]
        psarbull = [None] * length
        psarbear = [None] * length
        bull = True
        af = iaf
        hp = high[0]
        lp = low[0]

        for i in range(2,length):
            if bull:
                psar[i] = psar[i - 1] + af * (hp - psar[i - 1])
            else:
                psar[i] = psar[i - 1] + af * (lp - psar[i - 1])
            reverse = False
            if bull:
                if low[i] < psar[i]:
                    bull = False
                    reverse = True
                    psar[i] = hp
                    lp = low[i]
                    af = iaf
            else:
                if high[i] > psar[i]:
                    bull = True
                    reverse = True
                    psar[i] = lp
                    hp = high[i]
                    af = iaf
            if not reverse:
                if bull:
                    if high[i] > hp:
                        hp = high[i]
                        af = min(af + iaf, maxaf)
                    if low[i - 1] < psar[i]:
                        psar[i] = low[i - 1]
                    if low[i - 2] < psar[i]:
                        psar[i] = low[i - 2]
                else:
                    if low[i] < lp:
                        lp = low[i]
                        af = min(af + iaf, maxaf)
                    if high[i - 1] > psar[i]:
                        psar[i] = high[i - 1]
                    if high[i - 2] > psar[i]:
                        psar[i] = high[i - 2] 
            if bull:
                psarbull[i] = psar[i]
            else:
                psarbear[i] = psar[i]

        self.data["Parabolic_SAR"] = np.nan
        self.data["SAR_Bear"] = np.nan
        self.data["SAR_Bull"] = np.nan
        self.data["Parabolic_SAR"] = psar
        self.data["SAR_Bear"] = psarbear
        self.data["SAR_Bull"] = psarbull

        return
    
    def set_bbands_nk(self, n, k):
        self.bbands_nk = [n, k]
        return
    
    def add_bollinger_bands(self):
        self.bbands = True
        # print('BBands added')

        n = self.bbands_nk[0]
        k = self.bbands_nk[1]
        middle_bb = self.data.Close.rolling(n).mean()
        if ('MA_' + str(n)) not in self.data.columns:
            self.data['MA_' + str(n)] = middle_bb
        
        n_std = self.data.Close.rolling(n).std()
        upper_bb = middle_bb + k * n_std
        lower_bb = middle_bb - k * n_std

        self.data['Upper_BBand'] = upper_bb
        self.data['Lower_BBand'] = lower_bb

        return

    def add_stochastic_oscillator(self):
        self.kd = True

        length = len(self.data)
        if length < 8:
            return
        high = list(self.data.High)
        low = list(self.data.Low)
        close = list(self.data.Close)
        rsv = [None] * length
        k = [50] * length
        d = [50] * length
        for i in range(8,length):
            today_close = close[i]
            recent_high = max(high[i-8:i+1])
            recent_low = min(low[i-8:i+1])
            rsv[i] = 100 * ((today_close - recent_low) / (recent_high - recent_low))
            k[i] = (k[i-1] * 2 / 3) + (rsv[i] * 1 / 3)
            d[i] = (d[i-1] * 2 / 3) + (k[i] * 1 / 3)

        self.data['%K'] = np.nan
        self.data['%D'] = np.nan
        self.data['%K'] = k
        self.data['%D'] = d

        return

    def add_macd(self):
        self.macd = True

        self.data['EMA_12'] = self.data.Close.ewm(span=12, adjust=False).mean()
        self.data['EMA_26'] = self.data.Close.ewm(span=26, adjust=False).mean()
        self.data['DIF'] = self.data['EMA_12'] - self.data['EMA_26']
        self.data['MACD'] = self.data.DIF.ewm(span=9, adjust=False).mean()
        self.data['DIF-MACD'] = self.data['DIF'] - self.data['MACD']

        return

    def set_account_level(self, level):
        self.account_level = level
        self.current_level = [level, level]

        return

    def plot_ohlc(self):
        data = []

        number_of_graphs = 2
        if self.macd == True:
            number_of_graphs += 1

        if self.kd == True:
            number_of_graphs += 1
        
        if number_of_graphs == 2:
            set_yaxis_4 = 'y2'
            set_yaxis_3 = 'y1'
            set_yaxis_2 = None
            set_yaxis_1 = None
        elif number_of_graphs == 3:
            set_yaxis_4 = 'y3'
            set_yaxis_3 = 'y2'
            set_yaxis_2 = 'y1'
            set_yaxis_1 = 'y1'
        elif number_of_graphs == 4:
            set_yaxis_4 = 'y4'
            set_yaxis_3 = 'y3'
            set_yaxis_2 = 'y2'
            set_yaxis_1 = 'y1'

        if self.macd == True:
            trace_dif = go.Scatter(
                x=self.data.index, 
                y=self.data['DIF'],
                mode='lines',
                line=dict(width=2),
                marker=dict(color='#3D9970'),
                name='DIF',
                yaxis=set_yaxis_1)
            data.append(trace_dif)

            trace_macd = go.Scatter(
                x=self.data.index, 
                y=self.data['MACD'],
                mode='lines',
                line=dict(width=2),
                marker=dict(color='#FF4136'),
                name='MACD',
                yaxis=set_yaxis_1)
            data.append(trace_macd)

        if self.kd == True:
            trace_k = go.Scatter(
                x=self.data.index, 
                y=self.data['%K'],
                mode='lines',
                line=dict(width=2),
                marker=dict(color='#3D9970'),
                name='%K',
                yaxis=set_yaxis_2)
            data.append(trace_k)

            trace_d = go.Scatter(
                x=self.data.index, 
                y=self.data['%D'],
                mode='lines',
                line=dict(width=2),
                marker=dict(color='#FF4136'),
                name='%D',
                yaxis=set_yaxis_2)
            data.append(trace_d)

        if self.bbands == True:
            trace_upper_bb = go.Scatter(
                x=self.data.index, 
                y=self.data.Upper_BBand,
                mode='lines',
                line=dict(width=2),
                marker=dict(color='#ccc'),
                hoverinfo='none',
                legendgroup='Bollinger Bands',
                name='Bollinger Bands',
                yaxis=set_yaxis_4)
            data.append(trace_upper_bb)

            trace_lower_bb = go.Scatter(
                x=self.data.index, 
                y=self.data.Lower_BBand,
                mode='lines',
                line=dict(width=2),
                marker=dict(color='#ccc'),
                hoverinfo='none',
                legendgroup='Bollinger Bands',
                showlegend=False,
                yaxis=set_yaxis_4)
            data.append(trace_lower_bb)

        if self.sar == True:
            trace_bull = go.Scatter(
                x=self.data.index, 
                y=self.data.SAR_Bull,
                mode='markers',
                marker=dict(size=2, color='#3D9970'),
                name='Parabolic SAR',
                hoverinfo='none',
                legendgroup='Parabolic SAR',
                yaxis=set_yaxis_4)
            data.append(trace_bull)

            trace_bear = go.Scatter(
                x=self.data.index, 
                y=self.data.SAR_Bear,
                mode='markers',
                marker=dict(size=2, color='#FF4136'),
                hoverinfo='none',
                legendgroup='Parabolic SAR',
                showlegend=False,
                yaxis=set_yaxis_4)
            data.append(trace_bear)

        if self.ma == True:
            trace_ma_1 = go.Scatter(
                x=self.data.index, 
                y=self.data['MA_' + str(self.ma_period[0])],
                mode='lines',
                line=dict(width=2),
                marker=dict(color='#FFB11B'),
                name='MA_' + str(self.ma_period[0]),
                yaxis=set_yaxis_4)
            data.append(trace_ma_1)
            
            trace_ma_2 = go.Scatter(
                x=self.data.index, 
                y=self.data['MA_' + str(self.ma_period[1])],
                mode='lines',
                line=dict(width=2),
                marker=dict(color='#622954'),
                name='MA_' + str(self.ma_period[1]),
                yaxis=set_yaxis_4)
            data.append(trace_ma_2)

            trace_ma_3 = go.Scatter(
                x=self.data.index, 
                y=self.data['MA_' + str(self.ma_period[2])],
                mode='lines',
                line=dict(width=2),
                marker=dict(color='#33A6B8'),
                name='MA_' + str(self.ma_period[2]),
                yaxis=set_yaxis_4)
            data.append(trace_ma_3)

        trace_ohlc = go.Candlestick(
            x=self.data.index,
            open=self.data.Open,
            high=self.data.High,
            low=self.data.Low,
            close=self.data.Close,
            name='OHLC',
            yaxis=set_yaxis_4)
        data.append(trace_ohlc)

        INCREASING_COLOR = '#3D9970'
        DECREASING_COLOR = '#FF4136'
        colors = []
        for i in range(len(self.data.Close)):
            if i != 0:
                if self.data.Close[i] > self.data.Close[i-1]:
                    colors.append(INCREASING_COLOR)
                else:
                    colors.append(DECREASING_COLOR)
            else:
                colors.append(DECREASING_COLOR)
        trace_volume = go.Bar(
            x=self.data.index, 
            y=self.data.Volume,
            marker=dict(color=colors),
            yaxis=set_yaxis_3,
            name='Volume'
        )
        data.append(trace_volume)

        if self.do_long == True:
            long_data = self.data.loc[self.data['Action'] == 'Long']
            trace_long = go.Scatter(
                x=long_data.index, 
                y=long_data.Price,
                mode='markers+text',
                marker=dict(size=12, color='#0000FF'),
                text = long_data["Action"],
                textposition ="bottom center", 
                name='Long',
                yaxis=set_yaxis_4)
            data.append(trace_long)
        
        if self.do_sell == True:
            sell_data = self.data.loc[self.data['Action'] == 'Sell']
            trace_sell = go.Scatter(
                x=sell_data.index, 
                y=sell_data.Price,
                mode='markers+text',
                marker=dict(size=12, color='#FF00FF'),
                text = sell_data["Action"],
                textposition ="bottom center", 
                name='Sell',
                yaxis=set_yaxis_4)
            data.append(trace_sell)
        
        fig = go.Figure(data=data)
        fig.layout.update(
            title=dict(
                text='[Backtest] ' + self.stock_id + ' ' + self.stock_name,
                x=0.05),
            legend = dict(
                orientation = "h",
                x = 0,
                y = 1.05,
                yanchor = "top"
            ),
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                            label="1m",
                            step="month",
                            stepmode="backward"),
                        dict(count=3,
                            label="3m",
                            step="month",
                            stepmode="backward"),
                        dict(count=6,
                            label="6m",
                            step="month",
                            stepmode="backward"),
                        dict(count=1,
                            label="1y",
                            step="year",
                            stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(
                    visible=True
                ),
                type="category"
            ),
            plot_bgcolor='rgb(250, 250, 250)',
            height = 1000
        )
        
        if number_of_graphs == 2:
            domain_2 = [0.15, 1]
            domain_1 = [0, 0.15]
            fig.layout.update(
                yaxis2=dict(
                    domain=domain_2,
                    autorange=True,
                    fixedrange=False
                ),
                yaxis=dict(
                    domain=domain_1,
                    showticklabels = False
                )
            )
        elif number_of_graphs == 3:
            domain_3 = [0.3, 1]
            domain_2 = [0.15, 0.3]
            domain_1 = [0, 0.15]
            fig.layout.update(
                yaxis3=dict(
                    domain=domain_3,
                    autorange=True,
                    fixedrange=False
                ),
                yaxis2=dict(
                    domain=domain_2,
                    showticklabels = False
                ),
                yaxis=dict(
                    domain=domain_1
                )
            )
        elif number_of_graphs == 4:
            domain_4 = [0.45, 1]
            domain_3 = [0.3, 0.45]
            domain_2 = [0.15, 0.3]
            domain_1 = [0, 0.15]
            fig.layout.update(
                yaxis4=dict(
                    domain=domain_4,
                    autorange=True,
                    fixedrange=False
                ),
                yaxis3=dict(
                    domain=domain_3,
                    showticklabels = False
                ),
                yaxis2=dict(
                    domain=domain_2
                ),
                yaxis=dict(
                    domain=domain_1
                )
            )

        # offline.plot(fig, filename = 'backtest.html', auto_open=False)
        fig_gain_loss = self.plot_gain_loss()
        self.merge_figures([fig, fig_gain_loss], 'backtest.html')
        
        print('Output plots to html file.')

        return

    def plot_gain_loss(self):
        each_win_lose = self.data.loc[(self.data["Win/Lose"] == "Win") | (self.data["Win/Lose"] == "Lose")]
        colors = []
        for i in range(len(each_win_lose)):
            if each_win_lose["Gain/Loss"].iloc[i] >= 0:
                colors.append("#3D9970")
            else:
                colors.append("#FF4136")

        # x_axis = each_win_lose["Datetime"].apply(lambda x: x.strftime("%Y/%m/%d %H:%M"))
        trace_each_action_by_datetime = go.Bar(
            x = each_win_lose.index,
            y = each_win_lose["Gain/Loss"],
            base = 0,
            marker = dict( color = colors ),
            name = "Gain/Loss in each trade"
        )

        trace_cumulate_gain_and_loss = go.Scatter(
            x = each_win_lose.index,
            y = each_win_lose["Gain/Loss"].cumsum(),
            # y = each_win_lose["Account"],
            mode = "lines",
            line = dict( color = "#622954" ),
            name = "Cummulated gain/loss after each trade"
        )
        
        fig = make_subplots(
            rows = 2, 
            cols = 1, 
            specs = [[{}], [{}]],
            shared_xaxes = True, 
            shared_yaxes = False,
            vertical_spacing = 0.001
        )

        fig.append_trace(trace_each_action_by_datetime, 1, 1)
        fig.append_trace(trace_cumulate_gain_and_loss, 2, 1)

        # Set styles
        fig["layout"].update(
            title = dict(
                text = "Gain and loss in each trade // Cummulated gain and loss after each trade",
                x = 0.05
            ),
            plot_bgcolor = "rgb(250, 250, 250)",
            height = 600,
            legend = dict(
                orientation = "h",
                x = 0,
                y = 1.1,
                yanchor = "top"
            ),
            # margin = dict(
            #     t = 40,
            #     b = 40,
            #     r = 40,
            #     l = 40
            # ),
            xaxis = dict(
                type = "category"
            ),
            yaxis = dict(
                domain = [0.5, 1]
            ),
            yaxis2 = dict(
                domain = [0, 0.5]
            ) 
        )
    
        return fig

    def merge_figures(self, figures, filename):
        dashboard = open(filename, 'w')
        dashboard.write("<html><head></head><body>" + "\n")
        add_js = True
        for fig in figures:
            inner_html = offline.plot(
                fig, include_plotlyjs=add_js, output_type='div'
            )
            dashboard.write(inner_html)
            add_js = False
        dashboard.write("</body></html>" + "\n")

    def output_csv(self):
        self.data.to_csv('./Backtest/result/result.csv')
        print('Output data to csv file.')

        return

    def calculate_stats(self, buy_sell, win_lose):
        times = len(self.data.loc[(self.data["Action"] == buy_sell) & (self.data["Win/Lose"] == win_lose)])
        data_filter = self.data[["Rate of Return", buy_sell]].loc[(self.data["Action"] == buy_sell) & (self.data["Win/Lose"] == win_lose)] 
        if sum(data_filter[buy_sell]) == 0:
            average = 0
        else:    
            average = (sum(data_filter["Rate of Return"]*data_filter[buy_sell])/sum(data_filter[buy_sell]))*100
        return [buy_sell, win_lose, times, average]

    def output_stats(self):
        # Part 1
        overview_array = []
        overview_array.append(["Action", "Win/Lose", "Times", "Average rate of return(%)"])
        overview_array.append(self.calculate_stats("Buy", "Win"))
        overview_array.append(self.calculate_stats("Buy", "Lose"))
        overview_array.append(self.calculate_stats("Sell", "Win"))
        overview_array.append(self.calculate_stats("Sell", "Lose"))
    
        # Print array
        overview_header = overview_array.pop(0)
        overview_array = pd.DataFrame(overview_array, columns=overview_header)

        # Part 2
        total_trade = len(self.data.loc[(self.data["Action"] == "Buy") | (self.data["Action"] == "Sell")])
        total_win = len(self.data.loc[self.data["Win/Lose"] == "Win"])
        total_lose = len(self.data.loc[self.data["Win/Lose"] == "Lose"])
        total_gain  = sum(self.data["Gain/Loss"].loc[self.data["Gain/Loss"] > 0 ])
        total_loss = sum(self.data["Gain/Loss"].loc[self.data["Gain/Loss"] < 0 ])
        self.profit = total_gain + total_loss
        if total_win == 0 or total_loss == 0 or total_lose == 0:
            profit_factor = "None"
        else:
            profit_factor = abs((total_gain/total_win)/(total_loss/total_lose))
        if total_trade == 0:
            average_win_rate = "None"
        else:
            average_win_rate = (total_win/total_trade)*100
        
        # Part 3
        max_gain = self.data["Gain/Loss"].max()
        max_loss = self.data["Gain/Loss"].min()  
        max_gain_rate = (self.data["Rate of Return"].max())*100
        max_loss_rate= (self.data["Rate of Return"].min())*100
        win_lose_streak = self.data["Win/Lose"].loc[(self.data["Win/Lose"] == "Win")|(self.data["Win/Lose"] == "Lose")]
        # Calculate max streak
        groups = groupby(win_lose_streak)
        result = [[label, sum(1 for _ in group)] for label, group in groups]
        result = pd.DataFrame(result, columns=["Win/Lose", "Streak"])
        max_gains = result["Streak"].loc[result["Win/Lose"] == "Win"].max()
        max_losses = result["Streak"].loc[result["Win/Lose"] == "Lose"].max()
    
        # Output txt
        with open("./Backtest/result/stats.txt", "w") as text_file:
            # Part 1 output
            print(overview_array, file=text_file)
            print("============================================================", file=text_file)
            # Part 2 output
            print("總出手次數: {}" .format(total_trade), file=text_file)
            print("總獲利次數: {}" .format(total_win), file=text_file)
            print("總虧損次數: {}" .format(total_lose), file=text_file)
            print("總出手次數: {}" .format(total_trade))
            print("總獲利次數: {}" .format(total_win))
            print("總虧損次數: {}" .format(total_lose))
            print("總淨利: {}" .format(self.profit), file=text_file)
            print("總淨利: {}" .format(self.profit))
            print("總獲利: {}" .format(total_gain), file=text_file)
            print("總虧損: {}" .format(total_loss), file=text_file)
            print("獲利因子(平均每筆獲利/平均每筆虧損): {}" .format(profit_factor), file=text_file)
            print("平均勝率(%): {}" .format(average_win_rate), file=text_file)
            print("============================================================", file=text_file)
            # Part 3 output
            print("歷史最大獲利: {}" .format(max_gain), file=text_file)
            print("歷史最大虧損: {}" .format(max_loss), file=text_file)
            print("歷史最大獲利率(%): {}" .format(max_gain_rate), file=text_file)
            print("歷史最大虧損率(%): {}" .format(max_loss_rate), file=text_file)
            print("最大連續獲利次數: {}" .format(max_gains), file=text_file)
            print("最大連續虧損次數: {}" .format(max_losses), file=text_file)  
        
        print('Output stats to txt file.')

    def output_result(self):
        self.plot_ohlc()
        self.output_stats()
        self.output_csv()

    def init_backtest(self):
        self.data['Action'] = np.nan
        self.data['Long'] = np.nan
        self.data['Short'] = np.nan
        self.data['Buy'] = np.nan
        self.data['Sell'] = np.nan
        self.data['Price'] = np.nan
        self.data['Quantity'] = np.nan
        self.data['Win/Lose'] = np.nan
        self.data['Gain/Loss'] = np.nan
        self.data['Rate of Return'] = np.nan
        self.data['Account'] = np.nan
        
        return

    def action_long(self, row):
        self.do_long = True
        long_price = self.data.Open.iloc[row]
        self.long_price = long_price
        long_quantity = 1
        self.quantity_on_hand = long_quantity
        capital = long_price * 1000 * long_quantity
        self.current_level[1] = self.current_level[0] - capital

        self.data.Action.iloc[row] = 'Long'
        self.data.Long.iloc[row] = 1
        self.data.Price.iloc[row] = long_price
        self.data.Quantity.iloc[row] = long_quantity
        self.data.Account.iloc[row] = self.current_level[1]

        self.tp_stop_gain = long_price * 1.35
        self.tp_stop_loss = long_price * 0.85
        
        return

    def action_sell(self, row):
        self.do_sell = True
        sell_price = self.data.Open.iloc[row]
        sell_quantity = self.quantity_on_hand
        self.quantity_on_hand = 0
        capital = sell_price * 1000 * sell_quantity
        self.current_level[1] = self.current_level[1] + capital

        self.data.Action.iloc[row] = 'Sell'
        self.data.Sell.iloc[row] = 1
        self.data.Price.iloc[row] = sell_price
        self.data.Quantity.iloc[row] = sell_quantity
        self.data.Account.iloc[row] = self.current_level[1]

        gain_loss = self.current_level[1] - self.current_level[0]
        self.data['Gain/Loss'].iloc[row] = gain_loss
        self.data['Rate of Return'].iloc[row] = (sell_price - self.long_price) / self.long_price
        self.long_price = 0
        if gain_loss >= 0:
            self.data['Win/Lose'].iloc[row] = 'Win'
        else:
            self.data['Win/Lose'].iloc[row] = 'Lose'

        self.current_level[0] = self.current_level[1]

        return

    def backtest_strategy_crossover(self):
        if self.get_data(self.stock_id) == False:
            return
        self.add_moving_average()
        self.add_bollinger_bands()
        self.init_backtest()

        to_long = True
        to_sell = False
        ma_short_period = 'MA_' + str(self.ma_period[0])
        ma_long_period = 'MA_' + str(self.ma_period[2])
        length = len(self.data)
        delay = 0
        for i in range(1,length-1):
            if delay > 0:
                delay = delay - 1
                continue

            if to_long:
                if (self.data[ma_short_period].iloc[i] >= self.data[ma_long_period].iloc[i]):
                    if (self.data[ma_short_period].iloc[i-1] < self.data[ma_long_period].iloc[i-1]):
                        to_long = False
                        to_sell = True
                        self.action_long(i+1)
                        delay = 5
            elif to_sell:
                if (self.data[ma_short_period].iloc[i] <= self.data[ma_long_period].iloc[i]):
                    if (self.data[ma_short_period].iloc[i-1] > self.data[ma_long_period].iloc[i-1]):
                        to_long = True
                        to_sell = False
                        self.action_sell(i+1)
                        delay = 5
                elif self.data.Close.iloc[i] >= self.tp_stop_gain:
                    to_long = True
                    to_sell = False
                    self.action_sell(i+1)
                    delay = 5
                elif self.data.Close.iloc[i] <= self.tp_stop_loss:
                    to_long = True
                    to_sell = False
                    self.action_sell(i+1)
                    delay = 5   

        self.output_result()

        return

    def backtest_strategy_sar_and_bbands(self):
        if self.get_data(self.stock_id) == False:
            return
        self.add_moving_average()
        self.add_parabolic_sar()
        self.add_bollinger_bands()
        self.init_backtest()

        to_long = True
        to_sell = False
        ma_short_period = 'MA_' + str(self.ma_period[0])
        ma_long_period = 'MA_' + str(self.ma_period[2])
        length = len(self.data)
        delay = 0
        for i in range(6,length-1):
            if delay > 0:
                delay = delay - 1
                continue

            if to_long:
                if (self.data['Parabolic_SAR'].iloc[i] <= self.data['Close'].iloc[i]):
                    if (self.data['Lower_BBand'].iloc[i-6] >= self.data['Low'].iloc[i-6] or
                    self.data['Lower_BBand'].iloc[i-5] >= self.data['Low'].iloc[i-5] or
                    self.data['Lower_BBand'].iloc[i-4] >= self.data['Low'].iloc[i-4] or
                    self.data['Lower_BBand'].iloc[i-3] >= self.data['Low'].iloc[i-3] or
                    self.data['Lower_BBand'].iloc[i-2] >= self.data['Low'].iloc[i-2] or
                    self.data['Lower_BBand'].iloc[i-1] >= self.data['Low'].iloc[i-1]):
                        to_long = False
                        to_sell = True
                        self.action_long(i+1)
                        delay = 5
            elif to_sell:
                if (self.data['Parabolic_SAR'].iloc[i] >= self.data['Close'].iloc[i]):
                    if (self.data['Upper_BBand'].iloc[i-6] <= self.data['High'].iloc[i-6] or
                    self.data['Upper_BBand'].iloc[i-5] <= self.data['High'].iloc[i-5] or
                    self.data['Upper_BBand'].iloc[i-4] <= self.data['High'].iloc[i-4] or
                    self.data['Upper_BBand'].iloc[i-3] <= self.data['High'].iloc[i-3] or
                    self.data['Upper_BBand'].iloc[i-2] <= self.data['High'].iloc[i-2] or
                    self.data['Upper_BBand'].iloc[i-1] <= self.data['High'].iloc[i-1]):
                        to_long = True
                        to_sell = False
                        self.action_sell(i+1)
                        delay = 5
                elif self.data.Close.iloc[i] >= self.tp_stop_gain:
                    to_long = True
                    to_sell = False
                    self.action_sell(i+1)
                    delay = 5
                elif self.data.Close.iloc[i] <= self.tp_stop_loss:
                    to_long = True
                    to_sell = False
                    self.action_sell(i+1)
                    delay = 5                 

        self.output_result()

        return

    def backtest_strategy_bbands(self):
        if self.get_data(self.stock_id) == False:
            return
        self.add_bollinger_bands()
        self.init_backtest()

        to_long = True
        to_sell = False
        length = len(self.data)
        delay = 0
        for i in range(14,length-1):
            if delay > 0:
                delay = delay - 1
                continue

            if to_long:
                if (self.data['Lower_BBand'].iloc[i] > self.data['Close'].iloc[i]):
                    to_long = False
                    to_sell = True
                    self.action_long(i+1)
                    delay = 5   
            elif to_sell:
                if (self.data['Upper_BBand'].iloc[i] < self.data['Close'].iloc[i]):
                    to_long = True
                    to_sell = False
                    self.action_sell(i+1)
                    delay = 5
                elif self.data.Close.iloc[i] >= self.tp_stop_gain:
                    to_long = True
                    to_sell = False
                    self.action_sell(i+1)
                    delay = 5
                elif self.data.Close.iloc[i] <= self.tp_stop_loss:
                    to_long = True
                    to_sell = False
                    self.action_sell(i+1)
                    delay = 5                 

        self.output_result()

        return

    def backtest_strategy_kd(self):
        if self.get_data(self.stock_id) == False:
            return
        self.add_stochastic_oscillator()
        self.init_backtest()

        to_long = True
        to_sell = False
        length = len(self.data)
        delay = 0
        for i in range(14,length-1):
            if delay > 0:
                delay = delay - 1
                continue

            if to_long:
                if (self.data['%K'].iloc[i] > self.data['%D'].iloc[i]):
                    if (self.data['%K'].iloc[i-1] < self.data['%D'].iloc[i-1]):
                        to_long = False
                        to_sell = True
                        self.action_long(i+1)
                        delay = 5   
            elif to_sell:
                if (self.data['%K'].iloc[i] < self.data['%D'].iloc[i]):
                    if (self.data['%K'].iloc[i-1] > self.data['%D'].iloc[i-1]):
                        to_long = True
                        to_sell = False
                        self.action_sell(i+1)
                        delay = 5
                elif self.data.Close.iloc[i] >= self.tp_stop_gain:
                    to_long = True
                    to_sell = False
                    self.action_sell(i+1)
                    delay = 5
                elif self.data.Close.iloc[i] <= self.tp_stop_loss:
                    to_long = True
                    to_sell = False
                    self.action_sell(i+1)
                    delay = 5                 

        self.output_result()

        return
    
    def backtest_strategy_macd(self):
        if self.get_data(self.stock_id) == False:
            return
        self.add_macd()
        self.init_backtest()

        to_long = True
        to_sell = False
        length = len(self.data)
        delay = 0
        for i in range(14,length-1):
            if delay > 0:
                delay = delay - 1
                continue

            if to_long:
                if (self.data['DIF-MACD'].iloc[i] > 0):
                    if (self.data['DIF-MACD'].iloc[i-1] > 0):
                        if (self.data['DIF-MACD'].iloc[i-2] > 0):
                            if (self.data['DIF-MACD'].iloc[i-3] > 0):
                                if (self.data['DIF-MACD'].iloc[i-4] > 0):
                                    if (self.data['DIF-MACD'].iloc[i-5] < 0):
                                        to_long = False
                                        to_sell = True
                                        self.action_long(i+1)
                                        delay = 5   
            elif to_sell:
                if (self.data['DIF-MACD'].iloc[i] < 0):
                    if (self.data['DIF-MACD'].iloc[i-1] < 0):
                        if (self.data['DIF-MACD'].iloc[i-2] < 0):
                            if (self.data['DIF-MACD'].iloc[i-3] < 0):
                                if (self.data['DIF-MACD'].iloc[i-4] < 0):
                                    if (self.data['DIF-MACD'].iloc[i-5] > 0):
                                        to_long = True
                                        to_sell = False
                                        self.action_sell(i+1)
                                        delay = 5
                elif self.data.Close.iloc[i] >= self.tp_stop_gain:
                    to_long = True
                    to_sell = False
                    self.action_sell(i+1)
                    delay = 5
                elif self.data.Close.iloc[i] <= self.tp_stop_loss:
                    to_long = True
                    to_sell = False
                    self.action_sell(i+1)
                    delay = 5                 

        self.output_result()

        return

    def apply_strategy_start(self, folder):
        print('System ' + folder + ' Started')
        print('__________', file=open("./" + folder + "/result/result.txt", "w"))
        self.stock_on_hand = pd.read_csv('./data/stock_on_hand.csv', encoding="cp950")
        self.company_list = pd.read_csv('./data/list.csv', encoding="cp950")

        return

    def apply_strategy_crossover(self):
        folder = 'Strategy_Crossover'
        self.apply_strategy_start(folder)
        for i in range(len(self.company_list)):
            stock_id = str(self.company_list['Id'].iloc[i])
            stock_name = self.company_list['Name'].iloc[i]
            self.get_data(stock_id)

            self.add_moving_average()
            ma_short_period = 'MA_' + str(self.ma_period[0])
            ma_long_period = 'MA_' + str(self.ma_period[2])

            this_stock = self.stock_on_hand.loc[self.stock_on_hand['Id'] == int(stock_id)]
            if this_stock['Price'].iloc[0] == 0:
                if self.data[ma_short_period].iloc[-1] >= self.data[ma_long_period].iloc[-1]:
                    if self.data[ma_short_period].iloc[-2] < self.data[ma_long_period].iloc[-2]:
                        print(stock_id + ' ' + stock_name + ' 短日線買入訊號')
                        print(stock_id + ' ' + stock_name + ' 短日線買入訊號', file=open("./" + folder + "/result/Result.txt", "a"))
            else:
                if self.data[ma_short_period].iloc[-1] <= self.data[ma_long_period].iloc[-1]:
                    if self.data[ma_short_period].iloc[-2] > self.data[ma_long_period].iloc[-2]:
                        print(stock_id + ' ' + stock_name + ' 短日線賣出訊號')
                        print(stock_id + ' ' + stock_name + ' 短日線賣出訊號', file=open("./" + folder + "/result/Result.txt", "a"))
                elif self.data['Close'].iloc[-1] >= (this_stock['Price'].iloc[0] * 1.25):
                    print(stock_id + ' ' + stock_name + ' 停利賣出訊號')
                    print(stock_id + ' ' + stock_name + ' 停利線賣出訊號', file=open("./" + folder + "/result/Result.txt", "a"))
                elif self.data['Close'].iloc[-1] <= (this_stock['Price'].iloc[0] * 0.9):
                    print(stock_id + ' ' + stock_name + ' 停損賣出訊號')
                    print(stock_id + ' ' + stock_name + ' 停損賣出訊號', file=open("./" + folder + "/result/Result.txt", "a"))
                
            self.data.to_csv('./data/' + stock_id + stock_name + '.csv')
        
        print('System Ended') 

        return

    def apply_strategy_sar_and_bbands(self):
        folder = 'Strategy_SAR_and_BBands'
        self.apply_strategy_start(folder)
        for i in range(len(self.company_list)):
            stock_id = str(self.company_list['Id'].iloc[i])
            stock_name = self.company_list['Name'].iloc[i]
            self.get_data(stock_id)
            
            self.add_moving_average()
            self.add_parabolic_sar()
            self.add_bollinger_bands()
            ma_short_period = 'MA_' + str(self.ma_period[0])
            ma_long_period = 'MA_' + str(self.ma_period[2])

            this_stock = self.stock_on_hand.loc[self.stock_on_hand['Id'] == int(stock_id)]
            if this_stock['Price'].iloc[0] == 0:
                if (self.data['Parabolic_SAR'].iloc[-1] <= self.data['Close'].iloc[-1]):
                    if (self.data['Lower_BBand'].iloc[-7] >= self.data['Low'].iloc[-7] or
                    self.data['Lower_BBand'].iloc[-6] >= self.data['Low'].iloc[-6] or
                    self.data['Lower_BBand'].iloc[-5] >= self.data['Low'].iloc[-5] or
                    self.data['Lower_BBand'].iloc[-4] >= self.data['Low'].iloc[-4] or
                    self.data['Lower_BBand'].iloc[-3] >= self.data['Low'].iloc[-3] or
                    self.data['Lower_BBand'].iloc[-2] >= self.data['Low'].iloc[-2]):
                        print(stock_id + ' ' + stock_name + ' SAR&BBands買入訊號')
                        print(stock_id + ' ' + stock_name + ' SAR&BBands買入訊號', file=open("./" + folder + "/result/Result.txt", "a"))
            else:
                if (self.data['Parabolic_SAR'].iloc[-1] >= self.data['Close'].iloc[-1]):
                    if (self.data['Upper_BBand'].iloc[-7] <= self.data['High'].iloc[-7] or
                    self.data['Upper_BBand'].iloc[-6] <= self.data['High'].iloc[-6] or
                    self.data['Upper_BBand'].iloc[-5] <= self.data['High'].iloc[-5] or
                    self.data['Upper_BBand'].iloc[-4] <= self.data['High'].iloc[-4] or
                    self.data['Upper_BBand'].iloc[-3] <= self.data['High'].iloc[-3] or
                    self.data['Upper_BBand'].iloc[-2] <= self.data['High'].iloc[-2]):
                        print(stock_id + ' ' + stock_name + ' SAR&BBands賣出訊號')
                        print(stock_id + ' ' + stock_name + ' SAR&BBands賣出訊號', file=open("./" + folder + "/result/Result.txt", "a"))
                elif self.data['Close'].iloc[-1] >= (this_stock['Price'].iloc[0] * 1.25):
                    print(stock_id + ' ' + stock_name + ' 停利賣出訊號')
                    print(stock_id + ' ' + stock_name + ' 停利線賣出訊號', file=open("./" + folder + "/result/Result.txt", "a"))
                elif self.data['Close'].iloc[-1] <= (this_stock['Price'].iloc[0] * 0.9):
                    print(stock_id + ' ' + stock_name + ' 停損賣出訊號')
                    print(stock_id + ' ' + stock_name + ' 停損賣出訊號', file=open("./" + folder + "/result/Result.txt", "a"))
                
            self.data.to_csv('./data/' + stock_id + stock_name + '.csv')
        
        print('System Ended') 

        return

    def apply_strategy_bbands(self):
        folder = 'Strategy_BBands'
        self.apply_strategy_start(folder)
        for i in range(len(self.company_list)):
            stock_id = str(self.company_list['Id'].iloc[i])
            stock_name = self.company_list['Name'].iloc[i]
            self.get_data(stock_id)
            
            self.add_bollinger_bands()

            this_stock = self.stock_on_hand.loc[self.stock_on_hand['Id'] == int(stock_id)]
            if this_stock['Price'].iloc[0] == 0:
                if (self.data['Lower_BBand'].iloc[-1] > self.data['Close'].iloc[-1]):
                    print(stock_id + ' ' + stock_name + ' BBands買入訊號')
                    print(stock_id + ' ' + stock_name + ' BBands買入訊號', file=open("./" + folder + "/result/Result.txt", "a"))
            else:
                if (self.data['Upper_BBand'].iloc[-1] < self.data['Close'].iloc[-1]):
                    print(stock_id + ' ' + stock_name + ' BBands賣出訊號')
                    print(stock_id + ' ' + stock_name + ' BBands賣出訊號', file=open("./" + folder + "/result/Result.txt", "a"))
                elif self.data['Close'].iloc[-1] >= (this_stock['Price'].iloc[0] * 1.25):
                    print(stock_id + ' ' + stock_name + ' 停利賣出訊號')
                    print(stock_id + ' ' + stock_name + ' 停利線賣出訊號', file=open("./" + folder + "/result/Result.txt", "a"))
                elif self.data['Close'].iloc[-1] <= (this_stock['Price'].iloc[0] * 0.9):
                    print(stock_id + ' ' + stock_name + ' 停損賣出訊號')
                    print(stock_id + ' ' + stock_name + ' 停損賣出訊號', file=open("./" + folder + "/result/Result.txt", "a"))
                
            self.data.to_csv('./data/' + stock_id + stock_name + '.csv')
        
        print('System Ended') 

        return
    
    def apply_strategy_kd(self):
        folder = 'Strategy_KD'
        self.apply_strategy_start(folder)
        for i in range(len(self.company_list)):
            stock_id = str(self.company_list['Id'].iloc[i])
            stock_name = self.company_list['Name'].iloc[i]
            self.get_data(stock_id)
            
            self.add_stochastic_oscillator()

            this_stock = self.stock_on_hand.loc[self.stock_on_hand['Id'] == int(stock_id)]
            if this_stock['Price'].iloc[0] == 0:
                if (self.data['%K'].iloc[-1] > self.data['%D'].iloc[-1]):
                    if (self.data['%K'].iloc[-2] < self.data['%D'].iloc[-2]):
                        print(stock_id + ' ' + stock_name + ' KD買入訊號')
                        print(stock_id + ' ' + stock_name + ' KD買入訊號', file=open("./" + folder + "/result/Result.txt", "a"))
            else:
                if (self.data['%K'].iloc[-1] < self.data['%D'].iloc[-1]):
                    if (self.data['%K'].iloc[-2] > self.data['%D'].iloc[-2]):
                        print(stock_id + ' ' + stock_name + ' KD賣出訊號')
                        print(stock_id + ' ' + stock_name + ' KD賣出訊號', file=open("./" + folder + "/result/Result.txt", "a"))
                elif self.data['Close'].iloc[-1] >= (this_stock['Price'].iloc[0] * 1.25):
                    print(stock_id + ' ' + stock_name + ' 停利賣出訊號')
                    print(stock_id + ' ' + stock_name + ' 停利線賣出訊號', file=open("./" + folder + "/result/Result.txt", "a"))
                elif self.data['Close'].iloc[-1] <= (this_stock['Price'].iloc[0] * 0.9):
                    print(stock_id + ' ' + stock_name + ' 停損賣出訊號')
                    print(stock_id + ' ' + stock_name + ' 停損賣出訊號', file=open("./" + folder + "/result/Result.txt", "a"))
                
            self.data.to_csv('./data/' + stock_id + stock_name + '.csv')
        
        print('System Ended') 

        return

    def apply_strategy_macd(self):
        folder = 'Strategy_MACD'
        self.apply_strategy_start(folder)
        for i in range(len(self.company_list)):
            stock_id = str(self.company_list['Id'].iloc[i])
            stock_name = self.company_list['Name'].iloc[i]
            self.get_data(stock_id)
            
            self.add_macd()

            this_stock = self.stock_on_hand.loc[self.stock_on_hand['Id'] == int(stock_id)]
            if this_stock['Price'].iloc[0] == 0:
                if (self.data['DIF-MACD'].iloc[-1] > 0):
                    if (self.data['DIF-MACD'].iloc[-2] > 0):
                        if (self.data['DIF-MACD'].iloc[-3] > 0):
                            if (self.data['DIF-MACD'].iloc[-4] > 0):
                                if (self.data['DIF-MACD'].iloc[-5] > 0):
                                    if (self.data['DIF-MACD'].iloc[-6] < 0):
                                        print(stock_id + ' ' + stock_name + ' MACD買入訊號')
                                        print(stock_id + ' ' + stock_name + ' MACD買入訊號', file=open("./" + folder + "/result/Result.txt", "a"))
            else:
                if (self.data['DIF-MACD'].iloc[-1] < 0):
                    if (self.data['DIF-MACD'].iloc[-2] < 0):
                        if (self.data['DIF-MACD'].iloc[-3] < 0):
                            if (self.data['DIF-MACD'].iloc[-4] < 0):
                                if (self.data['DIF-MACD'].iloc[-5] < 0):
                                    if (self.data['DIF-MACD'].iloc[-6] > 0):
                                        print(stock_id + ' ' + stock_name + ' MACD賣出訊號')
                                        print(stock_id + ' ' + stock_name + ' MACD賣出訊號', file=open("./" + folder + "/result/Result.txt", "a"))
                elif self.data['Close'].iloc[-1] >= (this_stock['Price'].iloc[0] * 1.25):
                    print(stock_id + ' ' + stock_name + ' 停利賣出訊號')
                    print(stock_id + ' ' + stock_name + ' 停利線賣出訊號', file=open("./" + folder + "/result/Result.txt", "a"))
                elif self.data['Close'].iloc[-1] <= (this_stock['Price'].iloc[0] * 0.9):
                    print(stock_id + ' ' + stock_name + ' 停損賣出訊號')
                    print(stock_id + ' ' + stock_name + ' 停損賣出訊號', file=open("./" + folder + "/result/Result.txt", "a"))
                
            self.data.to_csv('./data/' + stock_id + stock_name + '.csv')
        
        print('System Ended') 

        return
        
stock = stock_trader()
stock.set_backtest_stock_id_name('3035','智原')

# stock.get_data(stock.stock_id)
stock.backtest_strategy_crossover()

stock.add_moving_average()
stock.add_parabolic_sar()
stock.add_bollinger_bands()
stock.add_stochastic_oscillator()
stock.add_macd()
stock.plot_ohlc()
# stock.output_stats()
# stock.output_csv()

# stock.apply_strategy_crossover()
# stock.apply_strategy_sar_and_bbands()
# stock.apply_strategy_bbands()
# stock.apply_strategy_kd()
# stock.apply_strategy_macd()

#%%
class stock_trade():
    def __init__(self):
        self.sheet = []
        return
    
    def get_data(self, stock_id):
        startTime = date.today() - timedelta(days=100)
        endTime = date.today()
        return pdr.DataReader(stock_id + '.TW', 'yahoo', startTime, endTime)

    def apply_strategy_crossover(self):
        print('System Started')
        print('__________', file=open("./Strategy_Crossover/result/result.txt", "w"))
        stock_on_hand = pd.read_csv('./data/stock_on_hand.csv', encoding="cp950")
        company_list = pd.read_csv('./data/list.csv', encoding="cp950")
        for i in range(len(company_list)):
            stock_id = str(company_list['Id'].iloc[i])
            stock_name = company_list['Name'].iloc[i]
            df = self.get_data(stock_id)
            df['MA_5'] = df['Close'].rolling(5).mean()
            df['MA_20'] = df['Close'].rolling(20).mean()
            
            this_stock = stock_on_hand.loc[stock_on_hand['Id'] == int(stock_id)]
            if this_stock['Price'].iloc[0] == 0:
                if df['MA_5'].iloc[-1] >= df['MA_20'].iloc[-1]:
                    if df['MA_5'].iloc[-2] < df['MA_20'].iloc[-2]:
                        print(stock_id + ' ' + stock_name + ' 短日線買入訊號')
                        print(stock_id + ' ' + stock_name + ' 短日線買入訊號', file=open("./Strategy_Crossover/result/Result.txt", "a"))
            else:
                if df['MA_5'].iloc[-1] <= df['MA_20'].iloc[-1]:
                    if df['MA_5'].iloc[-2] > df['MA_20'].iloc[-2]:
                        print(stock_id + ' ' + stock_name + ' 短日線賣出訊號')
                        print(stock_id + ' ' + stock_name + ' 短日線賣出訊號', file=open("./Strategy_Crossover/result/Result.txt", "a"))
                elif df['Close'].iloc[-1] >= (this_stock['Price'].iloc[0] * 1.25):
                    print(stock_id + ' ' + stock_name + ' 停利賣出訊號')
                    print(stock_id + ' ' + stock_name + ' 停利線賣出訊號', file=open("./Strategy_Crossover/result/Result.txt", "a"))
                elif df['Close'].iloc[-1] <= (this_stock['Price'].iloc[0] * 0.9):
                    print(stock_id + ' ' + stock_name + ' 停損賣出訊號')
                    print(stock_id + ' ' + stock_name + ' 停損賣出訊號', file=open("./Strategy_Crossover/result/Result.txt", "a"))
                
            df.to_csv('./data/' + stock_id + stock_name + '.csv')

            row_info = [
            stock_id, 
            stock_name, 
            df.index[-1],
            df['High'].iloc[-1], 
            df['Low'].iloc[-1], 
            df['Open'].iloc[-1],
            df['Close'].iloc[-1],
            df['Volume'].iloc[-1], 
            df['Adj Close'].iloc[-1], 
            df['MA_5'].iloc[-1], 
            df['MA_20'].iloc[-1],
            df['MA_5'].iloc[-2], 
            df['MA_20'].iloc[-2]]
            self.sheet.append(row_info)
        
        self.sheet = pd.DataFrame(self.sheet, columns=[
            'Id','Name','Date','High','Low','Open','Close',
            'Volume','Adj Close','MA_5','MA_20','MA_5(yest)','MA_20(yest)'])
        
        self.sheet.to_csv('./Strategy_Crossover/result/trading_info.csv', encoding='utf_8_sig', index=False)
        print('System Ended') 
        return

# trade = stock_trade()
# trade.apply_stategy_ma_sar()


#%%
