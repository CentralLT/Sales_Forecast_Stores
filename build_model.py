import build_features
import pandas as pd
import numpy as np
import d6tflow

d6tflow.set_dir('../../Data/output/')
import sklearn
from sklearn.metrics import mean_squared_error
import cfg


class TaskBuildBaseline(d6tflow.tasks.TaskCSVPandas):
    """Task to create baseline model as suggested by course instructions
    week 2. Simply base forecast on values of month t-1"""

    def requires(self):
        # requires data on monthly level for October and yTest data set
        return {'input1': build_features.TaskTrainTestSplit()}

    def run(self):
        # Load yTest and item_cnt_month for October
        yVal = self.input()['input1']['yVal'].load()
        yTest = self.input()['input1']['yTest'].load()

        # Merge yTest data with October values, set NAN to 0 and clip(0,20)
        subBaseline = pd.merge(left=yTest, right=yVal, on=['shop_id', 'item_id'], how='left')
        subBaseline.rename(columns={'target': 'item_cnt_month'}, inplace=True)
        subBaseline.drop(columns=['shop_id', 'item_id'], inplace=True)
        subBaseline.fillna(0, inplace=True)
        subBaseline.loc[:, 'item_cnt_month'].clip(0, 20, inplace=True)

        self.save(subBaseline)


class TaskZeroPrediction(d6tflow.tasks.TaskCSVPandas):
    """Prediction for November with 0"""

    def requires(self):
        return build_features.TaskTrainTestSplit()

    def run(self):
        yTest = self.input()['yTest'].load()  # Load yTest data set
        yTest.drop(columns=['shop_id', 'item_id'], inplace=True)
        yTest.loc[:, 'item_cnt_month'] = 0  # set all observations to 0
        zeroPrediction = yTest
        self.save(zeroPrediction)


class TaskLinearModel(d6tflow.tasks.TaskCSVPandas):
    """Constructing linear Regression Model to predict November sales"""

    def requires(self):
        return build_features.TaskTrainTestSplit()

    def run(self):
        # Load train, val and test data

        XTrain = self.input()['XTrain'].load()
        yTrain = self.input()['yTrain'].load()
        XVal = self.input()['XVal'].load()
        yVal = self.input()['yVal'].load()
        XTest = self.input()['XTest'].load()
        yTest = self.input()['yTest'].load()

        from sklearn.linear_model import LinearRegression

        # Varlist for grid search of best fit model w.r.t. different variable combinations
        varlist = [cfg.Baseline, cfg.Lags, cfg.Shocks, cfg.Date, cfg.All]
        lr = LinearRegression(normalize=False)
        summary_table = pd.DataFrame(columns=['Model', 'RMSE'])
        models = ['Baseline', 'Lags', '%Shocks', 'Date', 'All']
        for j, i in enumerate(varlist):
            lr.fit(XTrain.loc[:, i], yTrain.target)
            ypred = lr.predict(XVal.loc[:, i])
            ypred = ypred.clip(0, 20)
            summary_table.loc[j] = [models[j], sklearn.metrics.mean_squared_error(yVal.target, ypred, squared=False)]
        print(summary_table)
        self.save(pd.DataFrame(summary_table))


class TaskLinearModelTimeHorizon(d6tflow.tasks.TaskCSVPandas):
    """Repeat analysis of TaskLinearModel but with different length of historic time horizon"""

    def requires(self):
        return build_features.TaskTrainTestSplit()

    def run(self):
        # Load train, val and test data

        XTrain = self.input()['XTrain'].load()
        yTrain = self.input()['yTrain'].load()
        XVal = self.input()['XVal'].load()
        yVal = self.input()['yVal'].load()
        XTest = self.input()['XTest'].load()
        yTest = self.input()['yTest'].load()

        from sklearn.linear_model import Ridge
        # Varlist for grid search of best fit model w.r.t. different variable combinations
        varlist = [cfg.Baseline, cfg.Lags, cfg.Shocks, cfg.Date, cfg.All]
        lengthHorizon = [12, 15, 18]  # length in months

        lr = Ridge(normalize=False)
        summary_table = pd.DataFrame(columns=['Time', 'Model', 'RMSE'])
        models = ['Baseline', 'Lags', 'Shocks', 'Date', 'All']
        k = 0  # Row counter for appending rows to dataframe
        for t in lengthHorizon:  # Loop over Time Horizon
            for j, i in enumerate(varlist):  # Loop over varlist
                lr.fit(XTrain[XTrain.index.values >= (33 - t)].loc[:, i],
                       yTrain[yTrain.index.values >= (33 - t)].target)
                ypred = lr.predict(XVal.loc[:, i])
                summary_table.loc[k, :] = (
                [t, models[j], sklearn.metrics.mean_squared_error(yVal.target, ypred.clip(0, 20), squared=False)])
                k += 1
        print(summary_table)
        self.save(pd.DataFrame(summary_table))


class TaskLinearModelSelect(d6tflow.tasks.TaskCSVPandas):
    """Prediction of best performing model and time horizon"""

    def requires(self):
        return build_features.TaskTrainTestSplit()

    def run(self):
        # Load train, val and test data

        XTrain = self.input()['XTrain'].load()
        yTrain = self.input()['yTrain'].load()
        XVal = self.input()['XVal'].load()
        yVal = self.input()['yVal'].load()
        XTest = self.input()['XTest'].load()
        yTest = self.input()['yTest'].load()

        from sklearn.linear_model import Ridge
        # Varlist for grid search of best fit model w.r.t. different variable combinations

        lr = Ridge(normalize=False, random_state=0)

        lr.fit(XTrain, yTrain.target)
        ypred = lr.predict(XTest)
        yTest.iloc[:, -1] = pd.Series(ypred.clip(0, 20))
        yTest = yTest.drop(columns='shop_id')
        yTest = yTest.rename(columns={'item_id': 'item_cnt_month'})
        yTest.fillna(0, inplace=True)

        self.save(pd.DataFrame(yTest))


class TaskRidgeModel(d6tflow.tasks.TaskCSVPandas):
    """Constructing Ridge Regression Model to predict November sales"""

    def requires(self):
        return build_features.TaskTrainTestSplit()

    def run(self):
        # Load train, val and test data

        XTrain = self.input()['XTrain'].load()
        yTrain = self.input()['yTrain'].load()
        XVal = self.input()['XVal'].load()
        yVal = self.input()['yVal'].load()
        XTest = self.input()['XTest'].load()
        yTest = self.input()['yTest'].load()

        from sklearn.linear_model import Ridge

        # Varlist for grid search of best fit model w.r.t. different variable combinations
        alpha_list = [0.01, 0.1, 1.0, 10]
        summary_table = pd.DataFrame(columns=['Model', 'RMSE'])
        models = ['0.01', '0.1', '1.0', '10']
        for j, i in enumerate(alpha_list):
            lr = Ridge(normalize=False, alpha=i)
            lr.fit(XTrain, yTrain.target)
            ypred = lr.predict(XVal)
            ypred = ypred.clip(0, 20)
            summary_table.loc[j] = [models[j], sklearn.metrics.mean_squared_error(yVal.target, ypred, squared=False)]

        self.save(summary_table)


class TaskRandomForest(d6tflow.tasks.TaskCSVPandas):
    """Random Forest Regressor with Hyperparameter search"""

    def requires(self):
        return build_features.TaskTrainTestSplit()

    def run(self):
        # Load train, val and test data

        XTrain = self.input()['XTrain'].load()
        yTrain = self.input()['yTrain'].load()
        XVal = self.input()['XVal'].load()
        yVal = self.input()['yVal'].load()
        XTest = self.input()['XTest'].load()
        yTest = self.input()['yTest'].load()

        from sklearn.ensemble import RandomForestRegressor

        summary_table = pd.DataFrame(columns=['estimator', 'feature', 'Depth', 'RMSE'])

        n_features = XTest.shape[1]
        E = [20, 50, 100]  # number of estimators
        MF = [int(0.25*n_features),  int(0.5*n_features), int(0.7*n_features)]  # max features
        MD = [i for i in range(5, 15, 5)]  # max depth

        c = 0  # Counter

        # Grid search over all hyperparameter combinations
        for i in E:
            for j in MF:
                for k in MD:
                    RF = RandomForestRegressor(random_state=0, n_estimators=i, max_depth=k,
                                               max_features=j)
                    RF.fit(XTrain, yTrain.target)
                    ypred = RF.predict(XVal)
                    ypred = ypred.clip(0, 20)
                    summary_table.loc[c, :] = ([i, j, c, sklearn.metrics.mean_squared_error(yVal.target, ypred, squared=False)])
                    k += 1

        self.save(summary_table)


class TaskKNN(d6tflow.tasks.TaskCSVPandas):
    """kNN Regressor with Hyperparameter search"""

    def requires(self):
        return build_features.TaskTrainTestSplit()

    def run(self):
        # Load train, val and test data
        XTrain = self.input()['XTrain'].load()
        yTrain = self.input()['yTrain'].load()
        XVal = self.input()['XVal'].load()
        yVal = self.input()['yVal'].load()
        XTest = self.input()['XTest'].load()
        yTest = self.input()['yTest'].load()

        from sklearn.neighbors import KNeighborsRegressor

        summary_table = pd.DataFrame(columns=['Neighbor', 'RMSE'])

        N = [5, 10, 15]  # number of neighbors
        k = 0  # Counter

        # Grid search over all hyperparameter combinations
        for i in N:
            model = KNeighborsRegressor(n_neighbors=i)
            model.fit(XTrain, yTrain.target)
            ypred = model.predict(XVal)
            ypred = ypred.clip(0, 20)
            summary_table.loc[k, :] = ([i, sklearn.metrics.mean_squared_error(yVal.target, ypred, squared=False)])
            k += 1

        self.save(summary_table)


class TaskDNN(d6tflow.tasks.TaskCSVPandas):

    def requires(self):
        return build_features.TaskTrainTestSplit()

    def run(self):
        # Load train, val and test data
        XTrain = self.input()['XTrain'].load()
        yTrain = self.input()['yTrain'].load()
        XVal = self.input()['XVal'].load()
        yVal = self.input()['yVal'].load()
        XTest = self.input()['XTest'].load()
        yTest = self.input()['yTest'].load()

        summary_table = pd.DataFrame(columns=['estimator', 'RMSE'])

        n_features = XTrain.shape[1]

        import tensorflow as tf

        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units = 90, activation='relu'),
            tf.keras.layers.Dense(units = 30, activation='relu'),
            tf.keras.layers.Dense(units = 1, activation='linear'),
        ])

        model.compile(optimizer='SGD', metrics='RootMeanSquaredError', loss='MSE')
        model.fit(x=XTrain, y=yTrain.target, epochs=10, batch_size=1000, validation_data=(XVal, yVal.target))
        ypred = model.predict(XVal)
        ypred = ypred.clip(0, 20)
        summary_table.loc[:,0] = sklearn.metrics.mean_squared_error(yVal.target, ypred, squared=False)
        print(sklearn.metrics.mean_squared_error(yVal.target, ypred, squared=False))

        self.save(summary_table)
