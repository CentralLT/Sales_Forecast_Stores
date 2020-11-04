import make_dataset
import pandas as pd
import numpy as np
import d6tflow
import downcast
import re
import calendar

d6tflow.set_dir('../../Data/output/')


class TaskRollingWindow(d6tflow.tasks.TaskPqPandas):
    """Rolling Window method to calculate rolling mean of target and price variable over the last x months"""

    def requires(self):
        return make_dataset.TaskMakeDataSet()

    def run(self):
        final = self.input().load()
        final.set_index(['date_block_num', 'shop_id', 'item_id'], inplace=True)
        from itertools import repeat
        l = [3]
        varlist = ['item_id_price', 'item_id_target', 'shop_id_target', 'shop_id_price', 'price', 'target']
        windows = [x for item in l for x in repeat(item, 6)]
        for var, window in zip(varlist, windows):
            rolling_mean = final[var].rolling(window=window).mean()
            final[var + str(window)] = rolling_mean

        return self.save(final)


class TaskDownCast(d6tflow.tasks.TaskPqPandas):
    """Merge date table and rolling window data set and downcast variable types
    from 64 to 32 bit to save space and time
    """

    def requires(self):
        return TaskMeanEncoding()

    def run(self):
        final = self.input().load()
        final = downcast.downcast_dtypes(final)

        return self.save(final)


class TaskLagStructure(d6tflow.tasks.TaskPqPandas):
    """Create lag structure of target and independent variables"""

    def requires(self):
        return {'input1': TaskRollingWindow(), 'input2': TaskCountWeekend()}

    def run(self):
        final = self.input()['input1'].load()



        index_cols = ['shop_id', 'item_id', 'date_block_num']
        final.set_index(index_cols, inplace=True)
        col_to_rename = list(final.columns.difference(final.index))
        final.reset_index(inplace=True)
        shift_range = [1, 2]

        for month_shift in shift_range:
            train_shift = final[index_cols + col_to_rename].copy()
            train_shift['date_block_num'] = train_shift['date_block_num'] + month_shift
            foo = lambda x: '{}_lag_{}'.format(x, month_shift) if x in col_to_rename else x
            train_shift = train_shift.rename(columns=foo)
            final = pd.merge(final, train_shift, on=index_cols, how='left').fillna(0)
            final = final.append(train_shift.loc[train_shift['date_block_num'] == 34], ignore_index=True).fillna(0)

        fit_cols = [col for col in final.columns if col[-1] in [str(item) for item in shift_range]]

        # to_drop_cols = list(set(list(final.columns)) - (set(fit_cols) | set(index_cols))) + [ 'date_block_num' ]
        dropcols = set(col_to_rename) - {'target'}
        final.drop(columns=dropcols, inplace=True)

        # Add Weekend variable to data set - fraction of weekends on overall days in month
        dateTable = self.input()['input2'].load()
        final = pd.merge(left=final, right=dateTable, on='date_block_num', how='left', validate='m:1')

        self.save(final)


class TaskCountWeekend(d6tflow.tasks.TaskPqPandas):
    """Counts number of Fridays, Saturdays and Sundays in a given month
    Number of weekend days related to sales count variable
    """

    def run(self):
        # Load train data on daily level
        train = pd.read_csv('../../Data/raw/sales_train.csv')

        # Create date table with date_block_num and date
        datetable = train.loc[ :, ['date', 'date_block_num']]

        # Transform date variable into datetime object
        datetable.date = pd.to_datetime(datetable.date, format='%d.%m.%Y')

        # Create variables for day, month and year
        datetable.loc[:, 'month'] = datetable.date.dt.month
        datetable.loc[:, 'day'] = datetable.date.dt.day
        datetable.loc[:, 'year'] = datetable.date.dt.year

        # Perform groupby to aggregate month and year per date_block
        datetable = datetable.groupby(by='date_block_num', as_index=False).agg({'month': 'max', 'year': 'max'})
        datetable2 = pd.DataFrame(columns=datetable.columns.values.tolist(), data=[[34, 11, 2015]])
        datetable = datetable.append(datetable2, ignore_index=True)

        # Count weekend days per month
        day = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dayIndex = [0, 1, 2, 3, 4, 5, 6]

        for k, l in zip(day, dayIndex):
            for i in range(datetable.shape[0]):
                datetable.loc[i, k] = sum([1 for j in calendar.monthcalendar(datetable.loc[i, 'year'],
                                           datetable.loc[i, 'month']) if j[l] != 0])

        datetable['weekend'] = (datetable['Friday'] + datetable['Saturday'] + datetable['Sunday']) / \
                                 (datetable['Friday'] + datetable['Saturday'] + datetable['Sunday'] +
                                  datetable['Monday'] + datetable['Tuesday'] + datetable['Wednesday'] + datetable['Thursday'])
        datetable.drop(columns=[ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'month', 'year' ],
                       inplace=True)
        return self.save(datetable)


class TaskNewMonthVar(d6tflow.tasks.TaskPqPandas):
    """Generating variables on monthly frequency for shop, item and item_category related data

    Input:
    final daily data - Merged data of shop, item, and item_category with daily sales_train data
    final monthly data - Merged data of shop, item, category and sales data with monthly frequency

    Output:
    final monthly data set - Monthly data by aggregating sales data (item_cnt and item_price) on a monthly frequency for
    multiple groups (shop_id, shop_group, item_id, item_group, item_category_id, category_group)
    """

    def requires(self):
        # requires merged data set
        return {'input1': TaskLagStructure(), 'input2': make_dataset.TaskGetShopData(),
                'input3': make_dataset.TaskGetItemData(), 'input4': make_dataset.TaskGetCategoryData()}

    def run(self):
        # Load monthly and daily data sets
        monthly = self.input()['input1'].load()

        # Load required data
        shops = self.input()['input2']['kmeansShop'].load()
        items = self.input()['input3']['kmeansItems'].load()
        categories = self.input()['input4']['kmeansCategory'].load()

        # Merge data using their unique identifiers
        monthly = pd.merge(left=monthly, right=items, on='item_id', how='left', validate='m:1')
        monthly = pd.merge(left=monthly, right=categories, on='item_category_id', how='left', validate='m:1')
        monthly = pd.merge(left=monthly, right=shops, on='shop_id', how='left', validate='m:1')

        monthly.drop(columns=['shop_name', 'item_name', 'item_category_name', 'cityname', 'locationname'], inplace=True)

        # Generate aggregate price and cnt variable group by variable in iterlist
        iterlist = [['date_block_num'], ['date_block_num', 'item_id'], ['date_block_num', 'shop_id'],
                    ['date_block_num', 'item_category_id' ], ['date_block_num', 'Item_Group'],
                    ['date_block_num', 'City_Group'], ['date_block_num', 'Cat_Group'], ['date_block_num', 'Location_Group']]
        indexlist = [['date_block_num'], ['date_block_num', 'item_id'], ['date_block_num', 'shop_id'],
                     ['date_block_num', 'item_category_id'], ['date_block_num', 'Item_Group'], ['date_block_num', 'City_Group' ],
                    ['date_block_num', 'Cat_Group'], ['date_block_num', 'Location_Group']]
        namelist = ['date', 'item_id', 'shop_id', 'item_cateory_id', 'Item_Group', 'City_Group', 'Cat_Group', 'Location_Group']

        for j, i in enumerate(iterlist):
            df_month_item = monthly.groupby(by=i, as_index=False).agg(
                {'target_lag_1': 'mean', 'target_lag_2': 'mean', 'target3_lag_1': 'mean'
                    , 'price_lag_1': 'mean', 'price_lag_2': 'mean'})
            colnames = {'target_lag_1': str(namelist[j]) + '_target_lag_1',
                        'target_lag_2': str(namelist[j]) + '_target_lag_2',
                        'target3_lag_1': str(namelist[j]) + '_target3_lag_1',
                        'price_lag_1': str(namelist[j]) + '_price_lag_1',
                        'price_lag_2': str(namelist[j]) + '_price_lag_2'}
            df_month_item.rename(columns=colnames, inplace=True)
            monthly = pd.merge(left=monthly, right=df_month_item, how='left', on=indexlist[j])
        monthly = monthly[monthly['date_block_num']> 11]

        self.save(monthly)


class TaskMeanEncoding(d6tflow.tasks.TaskPqPandas):
    """Mean Encoding of all Dummy Variables"""

    def requires(self):
        return TaskNewMonthVar()

    def run(self):

        final = self.input().load()
        items = pd.read_csv('../../Data/raw/items.csv')
        shops = pd.read_csv('../../Data/raw/shops.csv')


        final = pd.merge(left=final, right=items, on='item_id', how='left', validate='m:1')

        # Create New Variables Item-Store ID to create identifier for encoding
        final['itemstore'] = final['shop_id'].astype(str) + '_' + final['item_id'].astype(str)

        # Name Dit for encoded variables
        varlist = {'item_encode': 'item_id', 'category_encode': 'item_category_id', 'shop_encode': 'shop_id',
                   'city_encode': 'City_Group', 'date_encode': 'date_block_num', 'itemstore_encode': 'itemstore',
                   'location_encode': 'Location_Group', 'cat_encode': 'Cat_Group'}

        # Encoding w/o data leak / look-ahead bias
        for variable, var in varlist.items():
            cumsum = final.groupby(var)['target'].cumsum() - final['target']
            cumcnt = final.groupby(var)['target'].cumcount()
            encoded_feature = cumsum / cumcnt
            encoded_feature.fillna(final.target.mean(), inplace=True)
            final[str(variable)] = pd.DataFrame(cumsum)

        self.save(final)


class TaskLogChange(d6tflow.tasks.TaskPqPandas):
    """Creating variables to capture change in target variable and prices
    Time Series analysis shows trend in data and seasonality. Try to capture trend component by
    calculating absolute change in variables over time"""

    def requires(self):
        return TaskDownCast()

    def run(self):

        final = self.input().load()

        final['targetchg'] = final['target_lag_1'] - final['target_lag_2']
        final['pricechg'] = np.log(1 + final['price_lag_1']) - np.log(1 + final['price_lag_2'])

        '''lags = []
        for i in final.columns.values:
            if re.search('lag_2$', str(i)):
                lags += [i]
        final.drop(columns=lags, inplace=True)'''

        self.save(final)


class TaskTrainTestSplit(d6tflow.tasks.TaskPqPandas):
    """Task to split data into training, validation and test set based on
    date variable. Using two validation schemes: Classical hold out with only
    one month as validation set and forward validation with multiple time steps."""

    # Save multiple data sets, have to define persist variable that has to be equal to saved data
    persist = ['yTrain', 'XTrain', 'yVal', 'XVal', 'yTest', 'XTest']

    def requires(self):
        return TaskLogChange()

    def run(self):

        final = self.input().load()
        final.target = final.target.clip(0,20)

        # Test set
        mask = final.loc[:, 'date_block_num'] == 34
        XTest = final[mask].copy()
        yTest = pd.read_csv('../../Data/raw/test.csv')

        drop_var = ['shop_id', 'item_id', 'item_name', 'shop_name', 'item_category_id', 'city', 'target', 'itemstore',
                    'item_name', 'shop_name']

        # Train set
        mask = final.loc[:, 'date_block_num'] < 33
        XTrain = final[mask].copy()
        yTrain = XTrain.loc[:, ['target', 'date_block_num', 'shop_id', 'item_id']]

        XTrain.drop(columns=drop_var, inplace=True)
        XTest.drop(columns=drop_var, inplace=True)

        # Val set
        mask = final['date_block_num'] == 33
        XVal = final[mask].copy()
        yVal = XVal.loc[:, ['target', 'date_block_num', 'shop_id', 'item_id']]

        XVal.drop(columns = drop_var, inplace = True)

        date_var = ['date_block_num']

        XTrain.set_index(date_var, inplace=True)
        yTrain.set_index(date_var, inplace=True)
        XTest.set_index(date_var, inplace=True)
        XVal.set_index(date_var, inplace=True)
        yVal.set_index(date_var, inplace=True)

        self.save({'yTrain': yTrain, 'yTest': yTest, 'yVal': yVal,
                   'XTrain': XTrain, 'XTest': XTest, 'XVal': XVal})
