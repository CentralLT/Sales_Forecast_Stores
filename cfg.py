
# Variable Selection Linear Model

Baseline = ['price_lag_1', 'price3_lag_1',
            'target_lag_1', 'target3_lag_1',
            'tstd_lag_1']

Lags = ['item_id_price_lag_1', 'item_id_price3_lag_1',
       'item_id_target_lag_1', 'item_id_target3_lag_1',
       'item_id_tstd_lag_1', 'price_lag_1', 'price3_lag_1',
       'shop_id_price_lag_1', 'shop_id_price3_lag_1',
       'shop_id_target_lag_1', 'shop_id_target3_lag_1',
       'shop_id_tstd_lag_1', 'target_lag_1', 'target3_lag_1',
       'tstd_lag_1']

Shocks = ['item_id_price_lag_1', 'item_id_price3_lag_1',
       'item_id_target_lag_1', 'item_id_target3_lag_1',
       'item_id_tstd_lag_1', 'price_lag_1', 'price3_lag_1',
       'shop_id_price_lag_1', 'shop_id_price3_lag_1',
       'shop_id_target_lag_1', 'shop_id_target3_lag_1',
       'shop_id_tstd_lag_1', 'target_lag_1', 'target3_lag_1',
       'tstd_lag_1', 'targetchg', 'pricechg']

Date = ['item_id_price_lag_1', 'item_id_price3_lag_1',
       'item_id_target_lag_1', 'item_id_target3_lag_1',
       'item_id_tstd_lag_1', 'price_lag_1', 'price3_lag_1',
       'shop_id_price_lag_1', 'shop_id_price3_lag_1',
       'shop_id_target_lag_1', 'shop_id_target3_lag_1',
       'shop_id_tstd_lag_1', 'target_lag_1', 'target3_lag_1',
       'tstd_lag_1', 'date_encode', 'targetchg', 'pricechg']

All = ['item_id_price_lag_1', 'item_id_price3_lag_1',
       'item_id_target_lag_1', 'item_id_target3_lag_1',
       'item_id_tstd_lag_1', 'price_lag_1', 'price3_lag_1',
       'shop_id_price_lag_1', 'shop_id_price3_lag_1',
       'shop_id_target_lag_1', 'shop_id_target3_lag_1',
       'shop_id_tstd_lag_1', 'target_lag_1', 'target3_lag_1',
       'tstd_lag_1', 'item_encode', 'category_encode', 'shop_encode',
       'city_encode', 'date_encode', 'itemstore_encode', 'targetchg',
       'pricechg']

