from datetime import timedelta

from matplotlib import pyplot as plt
from pandas import DataFrame

from db import geo, products


def cut_date(pd, step_timedelta=timedelta(weeks=1), start_date=None,
             end_date=None):
    pd['date'] = pd['date'].astype('datetime64[ns]')
    pd.sort_values('date')
    if not start_date:
        start_date = pd.date[pd.index[0]]
    if not end_date:
        end_date = pd.date[pd.index[-1]]
    result_pd = []
    # print(start_date, end_date)
    # print(pd)
    d = start_date
    while start_date <= end_date:
        cut_date = start_date + step_timedelta
        pd_new = pd.loc[pd.date >= start_date]
        pd_new = pd_new.loc[pd_new.date < cut_date]

        if len(pd_new):
            result_pd.append(pd_new)

        # pd_new.to_csv(f'water/{start_date.f}')
        start_date = cut_date
    #
    # print(result_pd)
    # print(len(result_pd))
    return result_pd


def get_sku_by_type(type_id):
    water = products.filter_by_type(type_id)
    skus = water.SKU.to_list()
    return skus


def get_geo_clusters_by_city(city_id):
    frame = geo.pd.loc[geo.pd.cityId == city_id]
    return frame.geoCluster.to_list()


def plot_show(data_frame):
    data_frame.to_csv('water_in_geo.csv')
    data_frame.cumsum()
    data_frame.plot(x='date', y=['price', 'sales'])
    plt.show()


def set_mean_instead_of_null(
        data_frame: DataFrame,
        groped_column_name: str,
        fill_columns: list,
):
    groped_by_sku = data_frame.groupby(by=[groped_column_name]).mean()

    series_map = {column_name: [] for column_name in fill_columns}

    for index, series in groped_by_sku.iterrows():
        for column in series_map:
            temp_df = data_frame.loc[data_frame[groped_column_name] == index]
            mean_value = groped_by_sku[column][index]
            fill_series = temp_df[column].fillna(mean_value)
            series_map[column] = series_map[column] + fill_series.to_list()

    for column_name, series in series_map.items():
        data_frame = data_frame.drop(columns=[column_name])
        data_frame[column_name] = series

    return data_frame

