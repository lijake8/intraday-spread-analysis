# dataset downloaded from http://www.kibot.com/Support.aspx#aggregate_bidask_data_format
# http://api.kibot.com/?action=history&symbol=IVE&interval=tickbidask&bp=1&user=guest
# ETF: IVE: S&P 500 Value ETF

# imports
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# Class to organize data and functions
class Analyzer:
    # constructor
    def __init__(self, filename):
        # read in the data
        self.df = pd.read_csv(filename, sep=',', header=None)
        self.df.columns = ['Date', 'Time', 'Price', 'Bid', 'Ask', 'Volume']
        
        self.df_small, self.df_volatility = None, None

        self.intervals = [1, 5, 10, 15, 20, 30, 60]

        # colorscales
        # Create the viridis colormap
        viridis_hex = ['#440154', '#482878', '#3E4A89', '#31688E', '#26828E', '#1F9E89', '#35B779', '#6DCD59', '#B4DE2C', '#FDE725']
        viridis = LinearSegmentedColormap.from_list('viridis', viridis_hex)
        # Interpolate the colormap to have 13 colors (for 30 minute intervals)
        new_viridis = viridis(np.linspace(0, 1, 13))
        # Convert the interpolated colors to hex values
        self.viridis_colorscale = [colors.rgb2hex(rgb) for rgb in new_viridis]
        self.default_colorscale = ['#636EFA'] * 15

        # display original data
        print('OG data has {} trading days. Sample:'.format(len(pd.unique(self.df['Date']))))
        print(self.df.head())

    # perform initial data cleaning
    def initial_clean(self):
        # convert date and time columns to datetime format
        self.df['Time'] = pd.to_datetime(self.df['Time'], format='%H:%M:%S')
        self.df['Time_pretty'] = self.df['Time'].dt.time
        self.df['Date'] = pd.to_datetime(self.df['Date'], format='%m/%d/%Y')

        # extract year and month into new columns
        self.df['Year'] = pd.DatetimeIndex(self.df['Date']).year
        self.df['Month'] = pd.DatetimeIndex(self.df['Date']).month

        # remove rows with missing data and 0's for the prices
        self.df = self.df[self.df['Bid'] > 0]
        self.df = self.df.dropna()

        # remove rows with time before 9:30am and after 4:00pm
        self.df = self.df[self.df['Time'] >= pd.to_datetime('09:30:00', format='%H:%M:%S')]
        self.df = self.df[self.df['Time'] < pd.to_datetime('16:00:00', format='%H:%M:%S')] # < not <= because the interval bins would only have 1 type of entry for the bin starting at 4:00

    # create calculated fields
    def calculated_fields(self):
        # insert columns for spread
        self.df['Spread'] = self.df['Ask'] - self.df['Bid']
        self.df['Spread_as_Pct'] = self.df['Spread'] / self.df['Price'] * 100 # in percent

        # find the max spread in the dataset
        max_spread = self.df['Spread_as_Pct'].max()
        print(f'The max spread as % of price is {max_spread}, which means we are going to remove problematic rows')

        # remove rows with bad data, e.g. spread % is more than 2% of the price
        self.df = self.df[self.df['Spread_as_Pct'] < 2]

    # filter into smaller dataframe
    def smaller_df(self):
        # create smaller dataframe with only the columns we need
        self.df_small = self.df.drop(['Date'], inplace=False, axis=1)

        # reorder columns
        cols_order = ['Year', 'Month', 'Time', 'Time_pretty', 'Bid', 'Ask', 'Price', 'Spread', 'Spread_as_Pct', 'Volume']
        self.df_small = self.df_small[cols_order]

    # returns a dataframe of volatilities for a given year/month
    def get_volatility_df(self, factor):
        """
        factor: 'Year' or 'Month'
        """
        periods = 252 if factor == 'Year' else 21 # 252 trading days in a year, 21 trading days in a month
        
        # get daily prices (open)
        self.df_daily_prices = self.df.groupby(['Date'])['Price', 'Year', 'Month'].first().reset_index()
        self.df_daily_prices.rename(columns={'Price': 'Open_price'}, inplace=True)

        # create a column for log of quotient of open prices
        self.df_daily_prices['Log_return'] = np.log(self.df_daily_prices['Open_price'] / self.df_daily_prices['Open_price'].shift(1))
        self.df_daily_prices = self.df_daily_prices.dropna()

        # perform grouping by standard deviation to get volatility
        self.df_volatility = self.df_daily_prices.groupby([factor])['Log_return'].std().reset_index()
        self.df_volatility.rename(columns={'Log_return': 'Std_dev_of_returns'}, inplace=True)
        self.df_volatility['Volatility'] = self.df_volatility['Std_dev_of_returns'] * np.sqrt(periods) # annualize the volatility
        self.df_volatility.drop(['Std_dev_of_returns'], inplace=True, axis=1)

        return self.df_volatility

    def double_line_graph(self, df_merged, interval):
        # line graph of spread and volume on 2 y axes against time
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=df_merged['Time_pretty'], y=df_merged['Spread_as_Pct'], name='Spread as % of Avg Price', marker_color='#636EFA'), secondary_y=False)
        fig.add_trace(go.Scatter(x=df_merged['Time_pretty'], y=df_merged['Vol_as_pct_of_daily_vol'], name='Volume as % of Daily Volume', marker_color='#EE7674'), secondary_y=True)
        fig.update_layout(title=f'Spread and Volume in {interval} Minute Intervals Throughout Trading Hours', xaxis_title='Time', yaxis_title='Spread (as % of Price)', yaxis2_title='Volume (as % of Daily Volume)')
        fig.write_image(f'spread_plot_{interval}.png')

    def scatter_plot(self, df_merged, interval):
        # scatter plot with the spread and volume on x and y axes
        fig2 = px.scatter(df_merged, x='Vol_as_pct_of_daily_vol', y='Spread_as_Pct', color='Time_pretty', title=f'Spread and Volume in {interval} Minute Intervals Throughout Trading Hours', color_discrete_sequence=self.viridis_colorscale, trendline='ols')
        fig2.update_traces(marker_size=10)
        fig2.update_layout(title=f'Spread and Volume Correspondence, {interval} Minute Intervals', xaxis_title='Volume (as % of Daily Volume)', yaxis_title='Spread (as % of Price)', legend_title_text='Time')

        # show legend entries by hour
        for i, _ in enumerate(fig2['data']): 
            if fig2['data'][i]['legendgroup'][-2:] == '00':
                fig2['data'][i]['showlegend'] = True
            else:
                fig2['data'][i]['showlegend'] = False

        fig2.write_image(f'scatter_plot_{interval}.png')

    # function that generates a grouped bar chart based on a macro factor (e.g. year, month, etc.)
    def grouped_plot(self, factor, interval, label):
        """
        factor: 'Year' or 'Month'
        interval: some number of minutes to group by, e.g. 1, 5, 10, 15, 20, 30, 60
        """
        # get corresponding total volumes for this interval schema
        df_vol_grouped_by_factor = self.df_small.groupby([factor, label])['Volume'].sum().reset_index()

        # normalize the volume to a % of total daily volume
        df_vol_grouped_by_factor['Vol_as_pct_of_daily_vol'] = df_vol_grouped_by_factor['Volume'] / df_vol_grouped_by_factor['Volume'].sum() * 100 # in percent
        df_vol_grouped_by_factor['Time_pretty'] = df_vol_grouped_by_factor[label].dt.time

        # create a new dataframe with the average spread for each interval bin
        df_spread_grouped_by_factor = self.df_small.groupby([factor, label])['Spread', 'Spread_as_Pct'].mean().reset_index()

        # merge the two dataframes to bring spread and volume together based on interval and month
        df_merged_by_factor = pd.merge(df_spread_grouped_by_factor, df_vol_grouped_by_factor, on=[factor, label])
        df_merged_by_factor.drop(['Volume', label], inplace=True, axis=1)
        
        # convert the time column to a string for graphing
        df_merged_by_factor['Time_pretty'] = df_merged_by_factor['Time_pretty'].astype(str)
        df_merged_by_factor['Time_pretty'] = df_merged_by_factor['Time_pretty'].str.slice(0, 5)
        
        # plot the grouped bar chart
        fig4 = make_subplots(specs=[[{"secondary_y": True}]])
        fig3 = px.histogram(df_merged_by_factor, x=factor, y='Spread_as_Pct', color='Time_pretty', barmode='group', color_discrete_sequence=self.default_colorscale)        

        # add traces from plotly express histogram to subplot figure
        for trace in fig3.select_traces():
            fig4.add_trace(trace, secondary_y = False)

        # handle data for and add secondary axis
        df_volatility = self.get_volatility_df(factor)
        fig4.add_trace(go.Scatter(x=df_volatility[factor], y=(df_volatility['Volatility'] * 100), mode='lines', marker_color='#EE7674', showlegend=True), secondary_y=True) 
        fig4.update_layout(title=f'Average Spread in {interval} Minute Intervals Throughout Trading Hours by {factor}', xaxis_title=factor, yaxis_title='Spread (as % of Price)', yaxis2_title='Volatility (%)', xaxis={"dtick":1}, barmode='group')
        
        # show legend only for the second trace
        for i, _ in enumerate(fig4['data']): 
            fig4['data'][i]['showlegend'] = False
        fig4['data'][-1]['showlegend'] = True
        fig4['data'][-1]['name'] = 'Volatility (%)'
        fig4['data'][-2]['showlegend'] = True
        fig4['data'][-2]['name'] = 'Spread (% of Price)'

        fig4.write_image(f'spread_plot_{factor}ly.png')

    # get the correlation and p-value between volume and spread
    def stat_analysis(self, df_merged):
        print('STATISTIC ANALYSIS')
        # get the pearson correlation coefficient and p-value
        print('Pearson correlation and p-val, all times:', pearsonr(df_merged['Spread_as_Pct'], df_merged['Vol_as_pct_of_daily_vol']))
        
        # repeat this for data before 2PM (14:00)
        df_merged_pre_2pm = df_merged[df_merged['Time_pretty'] < '14:00']
        print('Pearson correlation and p-val, pre-2PM data:', pearsonr(df_merged_pre_2pm['Spread_as_Pct'], df_merged_pre_2pm['Vol_as_pct_of_daily_vol']))

        # repeat this for data after 2PM (14:00)
        df_merged_post_2pm = df_merged[df_merged['Time_pretty'] >= '14:00']
        print('Pearson correlation and p-val, post-2PM data:', pearsonr(df_merged_post_2pm['Spread_as_Pct'], df_merged_post_2pm['Vol_as_pct_of_daily_vol']))

        # make a model to predict the spread based on volume alone
        # split the data into training and testing sets
        X = df_merged_pre_2pm['Vol_as_pct_of_daily_vol'].values.reshape(-1, 1)
        y = df_merged_pre_2pm['Spread_as_Pct'].values.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # create a linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        print('######################################################################################')
        print('Linear Regression Model')
        print('Coefficients: ', model.coef_)
        print('Intercept: ', model.intercept_)
        print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
        print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))
        print('######################################################################################')

    def analyze(self):
        for interval in self.intervals:
            print('\n\nFOR THE INTERVAL OF {} MINUTES:\n'.format(interval))

            # group the time into intervals bins (floor: 9:31am -> 9:30am, for example)
            label = f'Time_{interval}'
            self.df_small[label] = self.df_small['Time'].dt.floor(f'{interval}min')
            
            # get corresponding total volumes by interval bin
            self.df_vol = self.df_small.groupby([label])['Volume'].sum().reset_index()

            # normalize the volume to a % of total daily volume
            self.df_vol['Vol_as_pct_of_daily_vol'] = self.df_vol['Volume'] / self.df_vol['Volume'].sum() * 100 # in percent
            self.df_vol['Time_pretty'] = self.df_vol[label].dt.time

            # create a new dataframe with the average spread and % spread for each interval bin
            self.df_spread = self.df_small.groupby([label])['Spread', 'Spread_as_Pct'].mean().reset_index()

            # merge the dataframes to bring spread and volume together
            self.df_merged = pd.merge(self.df_spread, self.df_vol, on=label)
            self.df_merged.drop(['Volume', label], inplace=True, axis=1)
            
            # convert the time column to a string for graphing
            self.df_merged['Time_pretty'] = self.df_merged['Time_pretty'].astype(str)
            self.df_merged['Time_pretty'] = self.df_merged['Time_pretty'].str.slice(0, 5)

            # PLOT
            self.double_line_graph(self.df_merged, interval)
            self.scatter_plot(self.df_merged, interval)

            # STAT ANALYSIS
            self.stat_analysis(self.df_merged)

            # GROUPED PLOTS
            if interval == 20:
                self.grouped_plot('Month', interval, label)
                self.grouped_plot('Year', interval, label)


def main():
    IVE_analyzer = Analyzer("IVE_tickbidask.csv")
    IVE_analyzer.initial_clean()
    IVE_analyzer.calculated_fields()
    IVE_analyzer.smaller_df()
    IVE_analyzer.analyze()


if __name__ == "__main__":
    main()
