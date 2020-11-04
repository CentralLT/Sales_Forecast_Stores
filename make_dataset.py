import pandas as pd  # Data library to perform data manipulations in efficient way
import numpy as np  # Math library to perform matrix calculations in efficient way
import d6tflow  # Dataflow management tool
import os  # Basic os library of Python to perform OS commands in Python console
import re  # Basic Regex library of Python
from itertools import product  # Basic iterator product
import cfg
import word_process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import plotly.graph_objects as go
from scipy.sparse import *

d6tflow.set_dir('../../output/')

class TaskMakeDataSet(d6tflow.tasks.TaskPqPandas):
    """ Loads sales train data, expands data by generating all possible shop-item-date combinations, aggregating sales
    data on a monthly frequency and adding all item, shop and item_category identifiers

    Input:
        train_sales data - summarizes sales data (item_cnt_day and item_price) for all shops (shop_id) and items (item_id)
        on a daily basis
        item_category - category data after performing TaskGetCategoryData
        shop data - shop data after performing TaskGetShopData
        item data - item data after performing TaskGetItemData

    Output:
        final - sales data on a monthly frequency for item_cnt (sum of sold items per month in a specific store and stdv.
        of sales volume during month) and price (avg. price of item during that month in a specific store)
    """

    @property
    def run(self):

        train = pd.read_csv('../../Data/raw/sales_train.csv')

        train.drop(train[train.item_price <= 0].index.to_list(), axis=0, inplace=True)

        index_cols = ['shop_id', 'item_id', 'date_block_num']

        grid = []
        for block_num in train['date_block_num'].unique():
            cur_shops = train.loc[train['date_block_num'] == block_num, 'shop_id'].unique()
            cur_items = train.loc[train['date_block_num'] == block_num, 'item_id'].unique()
            grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])), dtype='int32'))

        grid = pd.DataFrame(np.vstack(grid), columns=index_cols, dtype=np.int32)

        gb = train.groupby(index_cols, as_index=False).agg({'item_cnt_day': ['sum', 'std'], 'item_price': 'mean'})
        gb.columns = [col[0] if col[-1] == '' else col[-1] for col in gb.columns.values]
        final = pd.merge(grid, gb, how='left', on=index_cols).fillna(0)

        import re
        search = ['mean', 'sum', r'\bstd\b']
        newname = ['price', 'target', 'tstd']
        for j in search:
            for i in final.columns.values:
                if re.search(j, str(i)):
                    final.rename(columns={i: newname[search.index(j)]}, inplace=True)

        gb = train.groupby(['shop_id', 'date_block_num'], as_index=False).agg(
            {'item_cnt_day': ['sum', 'std'], 'item_price': 'mean'})
        gb.columns = [col[0] if col[-1] == '' else col[-1] for col in gb.columns.values]
        final = pd.merge(final, gb, how='left', on=['shop_id', 'date_block_num']).fillna(0)
        import re
        search = ['mean', 'sum', r'\bstd\b']
        newname = ['shop_id_price', 'shop_id_target', 'shop_id_tstd']
        for j in search:
            for i in final.columns.values:
                if re.search(j, str(i)):
                    final.rename(columns={i: newname[search.index(j)]}, inplace=True)

        gb = train.groupby(['item_id', 'date_block_num'], as_index=False).agg(
            {'item_cnt_day': ['sum', 'std'], 'item_price': 'mean'})
        gb.columns = [col[0] if col[-1]=='' else col[-1] for col in gb.columns.values]
        final = pd.merge(final, gb, how='left', on=['item_id', 'date_block_num']).fillna(0)

        import re
        search = ['mean', 'sum', r'\bstd\b']
        newname = ['item_id_price', 'item_id_target', 'item_id_tstd']
        for j in search:
            for i in final.columns.values:
                if re.search(j, str(i)):
                    final.rename(columns={i: newname[search.index(j)]}, inplace=True)

        return self.save(final)


class TaskGetShopData(d6tflow.tasks.TaskCSVPandas):
    """Load shops data set and perform simple NLP by splitting address line and identifying city of shop
    input: shops data set with shop_id and shop_name
    output: PCAshop - PCA components of Tfidf vector to visualize clustering in 2D space
            kmeans - Output of Kmeans clustering of shop_names based on shop_names"""

    persist = ['kmeansShop', 'Centers', 'features']

    def run(self):
        # Load data set
        shops = pd.read_csv('../../Data/raw/shops.csv')

        for j,i in enumerate(shops.shop_name):
            shops.loc[j,'cityname'] = list(i.split())[0]
            shops.loc[j,'locationname'] = ' '.join(list(i.split())[1:])


        # Preprocess shop_name data and tokenize data
        colslist = ['cityname', 'locationname']
        groupname = ['City_Group', 'Location_Group']
        Clusters = [14, 10]

        for i in [0, 1]:
            corpus = []
            for name in shops.loc[:, str(colslist[i])]:
                corpus.append(word_process.preprocess(name))

        # Construct TfidfVector of preprocessed data
            tfidf_vectorizer = TfidfVectorizer()
            feature_vector = tfidf_vectorizer.fit_transform(corpus).toarray()

        # Graph for Elbow method (plotting within variation for different number of clusters)
            distortions = []
            for K in range(1,30): # 60 different clusters (as we have 60 shops)
                kmeans = KMeans(n_clusters = K, random_state=0).fit(feature_vector)
                centers = pd.DataFrame(kmeans.cluster_centers_)
                distortions.append(sum(np.min(cdist(feature_vector, kmeans.cluster_centers_, 'euclidean'), axis=1))/feature_vector.shape[0])

        # Plot the variation on y axis vs number of clusters on x axis
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=distortions, mode='lines', name='lines'))
            fig.show()

        # Cluster TfidifVector  (see graph)
            kmeans = KMeans(n_clusters=Clusters[i], random_state=0).fit(feature_vector)
            kmeansLabels = pd.DataFrame(kmeans.labels_)
            shops[groupname[i]] = kmeansLabels

        return self.save({ 'kmeansShop': shops, 'Centers': centers, 'features': pd.DataFrame(feature_vector)})


class TaskGetItemData(d6tflow.tasks.TaskCSVPandas):
    """Running TFIDF, PCA and Kmeans clustering to find clusters of Items based
    on item description

    input:
    - items data set with item_id, item_category_id and item_name

    output:
    - PCAitems: Output of PCA analysis of Tfidf vector to visualize item clustering in 2D space
    - kmeansitem: Output of kmeans clustering to identify groups of items based on item description
"""

    persist = ['features', 'kmeansItems']

    def run(self):

    # Load items data set
        items = pd.read_csv('../../Data/raw/items.csv')

    # Preprocess item description
        corpus = []
        for descr in items.item_name:
            corpus.append(word_process.preprocess(descr))

    # Construct Tfidf vector of preprocessed data
        tfidf_vectorizer = TfidfVectorizer()
        feature_vector = tfidf_vectorizer.fit_transform(corpus).toarray()

        S = coo_matrix(feature_vector)

    # Graph for Elbow method (plotting within variation for different number of clusters)
        distortions = []
        for K in range(1, 2):  # 60 different clusters (as we have 60 shops)
            kmeans = KMeans(n_clusters=K, random_state=0).fit(S.tocsr())
            centers = pd.DataFrame(kmeans.cluster_centers_)
            distortions.append(sum(np.min(cdist(feature_vector, kmeans.cluster_centers_, 'euclidean'), axis=1)) /
                               feature_vector.shape[0 ])

    # Plot the variation on y axis vs number of clusters on x axis
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=distortions, mode='lines', name='lines'))
        fig.show()


    # Cluster TfidifVector into 4 clusters (see graph)
        kmeans = KMeans(n_clusters=25, random_state=0).fit(S.tocsr())
        kmeansLabels = pd.DataFrame(kmeans.labels_)
        items['Item_Group'] = kmeansLabels

        self.save({'features': pd.DataFrame(feature_vector), 'kmeansItems': items})


class TaskGetCategoryData(d6tflow.tasks.TaskPqPandas):
    """Running TFIDF, PCA and Kmeans clustering to find clusters of Item_categories based
    on item_category description

    input:
    - items data set with item_category_id and item_category_name

    output:
    - PCAcategory: Output of PCA analysis of Tfidf vector to visualize item_category clustering in 2D space
    - kmeanscategory: Output of kmeans clustering to identify groups of item_categories based on item_category description
"""

    persist = ['kmeansCategory']

    def run(self):
        # Load items data set
        category = pd.read_csv('../../Data/raw/item_categories.csv')

        # Preprocess item description
        corpus = []
        for descr in category.item_category_name:
            newdescr = descr.split('(')
            newdescr = ' '.join(newdescr).split(')')
            newdescr = ' '.join(newdescr).split(',')
            newdescr = ' '.join(newdescr).split('-')
            corpus.append(word_process.preprocess(' '.join(newdescr)))

        # Construct Tfidf vector of preprocessed data
        tfidf_vectorizer = TfidfVectorizer()
        feature_vector = tfidf_vectorizer.fit_transform(corpus).toarray()

        # Graph for Elbow method (plotting within variation for different number of clusters)
        distortions = []
        for K in range(1, 60):  # 60 different clusters (as we have 60 shops)
            kmeans = KMeans(n_clusters=K, random_state=0).fit(feature_vector)
            distortions.append(sum(np.min(cdist(feature_vector, kmeans.cluster_centers_, 'euclidean'), axis=1)) /
                               feature_vector.shape[0])

        # Plot the variation on y axis vs number of clusters on x axis
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=distortions, mode='lines', name='lines'))
        fig.show()

        # Cluster TfidifVector into 4 clusters (see graph)
        kmeans = KMeans(n_clusters=12, random_state=0).fit(feature_vector)
        kmeansLabels = pd.DataFrame(kmeans.labels_)
        category['Cat_Group'] = kmeansLabels

        self.save({'kmeansCategory': category})


class TaskLagStructureNN(d6tflow.tasks.TaskPqPandas):
    """Create lag structure of target and independent variables"""

    def requires(self):
        return {'input1': TaskMakeDataSet()}

    def run(self):
        final = self.input()['input1'].load()
        drop_cols = ['item_id_price', 'item_id_target',
                     'price', 'shop_id_price', 'shop_id_target',
                     'shop_id_tstd', 'tstd', 'item_id_tstd']
        final.drop(columns= drop_cols, inplace=True)

        index_cols = ['shop_id', 'item_id', 'date_block_num']
        final.set_index(index_cols, inplace=True)
        col_to_rename = list(final.columns.difference(final.index))

        final.reset_index(inplace=True)


        for month_shift in range(35):
            train_shift = final[index_cols + col_to_rename].copy()
            train_shift['date_block_num'] = train_shift['date_block_num'] + month_shift
            foo = lambda x: '{}_lag_{}'.format(x, month_shift) if x in col_to_rename else x
            train_shift = train_shift.rename(columns=foo)
            final = pd.merge(final, train_shift, on=index_cols, how='left').fillna(0)
            final = final.append(train_shift.loc[train_shift['date_block_num'] == 34], ignore_index=True).fillna(0)

        fit_cols = [col for col in final.columns if col[-1] in [str(item) for item in range(35)]]

        # to_drop_cols = list(set(list(final.columns)) - (set(fit_cols) | set(index_cols))) + [ 'date_block_num' ]
        dropcols = set(col_to_rename) - {'target'}
        final.drop(columns=dropcols, inplace=True)

        self.save(final)


class TaskWideToLong(d6tflow.tasks.TaskPqPandas):

    def requires(self):
        return TaskMakeDataSet()

    def run(self):
        final = self.input().load()

        new = final.pivot(index=['shop_id', 'item_id'], columns='date_block_num', values='target')
        new = pd.DataFrame(new)
        new.columns = new.columns.values.astype(str)
        new.fillna(0, inplace=True)

        self.save(new)
