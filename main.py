from datetime import timedelta, datetime

from fastai.collab import tabular_learner
from fastai.data.transforms import Normalize
from fastai.metrics import accuracy
from fastai.tabular.core import Categorify, FillMissing
from fastai.tabular.data import TabularDataLoaders
from matplotlib import pyplot as plt
from pandas import DataFrame

from db import products, train, test
from split_date import SplitDate
from utils import cut_date
from validate import ValidateColumns

WATER_ID = 5550022
TROPIC_ID = 5551018
YOGURT_ID = 5550259

GEO_CLUSTER = 2053


def get_sku_by_type(TYPE_ID):
    water = products.filter_by_type(TYPE_ID)
    skus = water.SKU.to_list()
    return skus


def plot_show(data_frame):
    data_frame.to_csv('water_in_geo.csv')
    data_frame.cumsum()
    data_frame.plot(x='date', y=['price', 'sales'])
    plt.show()


def get_by_sku_type_id(data_frame, sku_type_id):
    skus = get_sku_by_type(sku_type_id)
    water_in = data_frame.loc[data_frame['SKU'].isin(skus)]
    return water_in


def get_by_cluster_id(data_frame, geo_cluster_id):
    data_frame = data_frame.loc[
        data_frame.geoCluster == geo_cluster_id].sort_values('date')
    return data_frame


def group_by_weeks():
    water = get_by_sku_type_id(train.pd, WATER_ID)
    water_in_geo = get_by_cluster_id(water, GEO_CLUSTER)

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


def test_learn():
    frame_skus_group_train = get_by_sku_type_id(train.pd, WATER_ID)
    # frame_skus_group_train = get_by_cluster_id(frame_skus_group_train, GEO_CLUSTER)

    frame_skus_group_test = get_by_sku_type_id(test.pd, WATER_ID)
    # frame_skus_group_test = get_by_cluster_id(frame_skus_group_test, GEO_CLUSTER)

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
    test_frame['sales'] = 1

    dls = TabularDataLoaders.from_df(
        learn_frame,
        y_names='sales',
        cat_names=[
            'SKU',
            'year',
            'geoCluster',
            'month',
            'day',
            'weekday'],
        cont_names=[
            'price',
            'sales'],
        procs=[Categorify, FillMissing, Normalize]
    )

    print(type(dls), dls)
    # dls.show_batch()

    learn = tabular_learner(dls, metrics=accuracy)
    learn.lr_find()
    learn.fit_one_cycle(10)
    # learn.show_results()

    dl = learn.dls.test_dl(test_frame)
    print(learn.get_preds(dl=dl))


if __name__ == '__main__':
    # print_hi('PyCharm')

    test_learn()
