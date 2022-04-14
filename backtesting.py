from trading import CommisionScheme, Strategy
import backtrader as bt
import backtrader.feeds as btfeeds
import pyfolio as pf
from datetime import datetime
import pandas as pd

price_path = './price.csv'
pos_path = './pos.csv'
# Initialize cerebro 
cerebro = bt.Cerebro()
cerebro.broker.setcash(1000000)
cerebro.broker.addcommissioninfo(CommisionScheme(commission=0.0004,automargin = 1))
bt_data = btfeeds.GenericCSVData(
    dataname=price_path,
    dtformat='%Y-%m-%d %H:%M:%S.%f',
    timeframe=bt.TimeFrame.MicroSeconds,
    datetime=0,
    high=1,
    low=2,
    open=3,
    close=4,
    volume=5,
    openinterest=-1
)

# Resample the data
# bt_data = cerebro.resampledata(bt_data, timeframe=bt.TimeFrame.MicroSeconds)

target = pd.read_csv(pos_path, index_col=0).to_dict().values()
print(target)
bt_data.target = target
cerebro.adddata(bt_data)

# Add a pyfolio analyzer to view the performance tearsheets
cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
cerebro.addstrategy(Strategy)

# Run startegy
start_time = datetime.now()
print('Starting Balance: %.2f' % cerebro.broker.getvalue())
results = cerebro.run()
ending_value = cerebro.broker.getvalue()
print(f'Final Portfolio Value: {ending_value:,.2f}')
print("--- %s seconds ---" % (datetime.now() - start_time))


pyfoliozer = results[0].analyzers.getbyname('pyfolio')
returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
pf.create_full_tear_sheet(
    returns,
    positions=positions,
    transactions=transactions,
    live_start_date='2021-10-31',  # This date is sample specific
    round_trips=True)

# backtrader plot
cerebro.plot()
figure = cerebro.plot(style='candlebars')[0][0]
figure.savefig(f'backtrader.png')# View Pyfolio results