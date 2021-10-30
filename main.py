from datetime import timedelta, datetime

from fastai.collab import tabular_learner
from fastai.data.transforms import Normalize, RandomSplitter
from fastai.metrics import accuracy, rmse
from fastai.tabular.core import Categorify, FillMissing, TabularPandas
from fastai.tabular.data import TabularDataLoaders
from fastai.tabular.learner import TabularLearner
from fastcore.basics import range_of
from matplotlib import pyplot as plt
from pandas import DataFrame, Series
from sklearn.metrics import r2_score

from db import products, train, test, geo
from split_date import SplitDate
from utils import cut_date
from validate import ValidateColumns

WATER_ID = 5550022
TROPIC_ID = 5551018
YOGURT_ID = 5550259

GEO_CLUSTER = 2053
CITY_ID = 1


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


def filter_by_sku_type_id(data_frame, sku_type_id):
    skus = get_sku_by_type(sku_type_id)
    frame = data_frame.loc[data_frame['SKU'].isin(skus)]
    return frame


def filter_by_cluster_id(data_frame, geo_cluster_id):
    data_frame = data_frame.loc[
        data_frame.geoCluster == geo_cluster_id].sort_values('date')
    return data_frame


def filter_by_city_id(data_frame, city_id):
    clusters = get_geo_clusters_by_city(city_id)
    frame = data_frame.loc[data_frame['geoCluster'].isin(clusters)]
    return frame


def group_by_weeks():
    water = filter_by_sku_type_id(train.pd, WATER_ID)
    water_in_geo = filter_by_cluster_id(water, GEO_CLUSTER)

    grouped_frame_1_week = cut_date(water_in_geo)
    grouped_frame_10_weeks = cut_date(
        water_in_geo,
        step_timedelta=timedelta(weeks=10)
    )
    return grouped_frame_1_week, grouped_frame_10_weeks


def clean_columns(data_frame):
    data_frame = data_frame.drop(columns=['ID'])
    # data_frame = data_frame.drop(columns=['geoCluster'])
    return data_frame


def set_mean_instead_of_null(data_frame: DataFrame):
    groped_by_sku = data_frame.groupby(by=['SKU']).mean()

    series_map = {
        'price': [],
        'sales': []
    }

    for index, series in groped_by_sku.iterrows():
        for column in series_map:
            temp_df = data_frame.loc[data_frame['SKU'] == index]
            mean_value = groped_by_sku[column][index]
            fill_series = temp_df[column].fillna(mean_value)
            series_map[column] = series_map[column] + fill_series.to_list()

    for column_name, series in series_map.items():
        data_frame = data_frame.drop(columns=[column_name])
        data_frame[column_name] = series

    return data_frame


def test_learn():
    # frame_skus_group_train = filter_by_sku_type_id(train.pd, WATER_ID)
    frame_skus_group_train = filter_by_city_id(train.pd, CITY_ID)
    frame_skus_group_train = set_mean_instead_of_null(frame_skus_group_train)

    # frame_skus_group_test = filter_by_sku_type_id(test.pd, WATER_ID)
    frame_skus_group_test = filter_by_city_id(test.pd, CITY_ID)

    learn_frame: DataFrame = frame_skus_group_train
    learn_frame = clean_columns(learn_frame)
    learn_frame = learn_frame.loc[learn_frame.price.notnull()]
    learn_frame = SplitDate(learn_frame).split()

    print(learn_frame.corr())
    print(learn_frame)

    test_frame: DataFrame = frame_skus_group_test
    test_frame = clean_columns(test_frame)
    test_frame = test_frame.loc[test_frame.price.notnull()]
    test_frame = SplitDate(test_frame).split()
    test_frame['sales'] = 0

    splits = RandomSplitter(valid_pct=0.2)(range_of(learn_frame))
    to = TabularPandas(learn_frame,
                       y_names='sales',
                       cat_names=[
                           'geoCluster',
                           'SKU',
                           'year',
                           'month',
                           'day',
                           'weekday'],
                       cont_names=[
                           'price',
                           'sales'],
                       procs=[Categorify, FillMissing],
                       splits=splits
                       )
    dls = to.dataloaders(2048)
    dls.show_batch()

    learn: TabularLearner = tabular_learner(
        dls,
        layers=[300, 200, 100, 50],
        model_dir='models',
        metrics=[rmse, r2_score])
    learn.lr_find(start_lr=1e-05, end_lr=1e+05, num_it=100)
    learn.fit_one_cycle(25)
    learn.save('stage 1')
    # learn.show_results()

    dl = learn.dls.test_dl(test_frame)
    preds, targs = learn.get_preds(dl=dl)
    print(preds, targs)


if __name__ == '__main__':
    # print_hi('PyCharm')

    test_learn()
