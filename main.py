from datetime import timedelta

from fastai.collab import tabular_learner
from fastai.data.transforms import Normalize
from fastai.metrics import accuracy
from fastai.tabular.core import Categorify, FillMissing
from fastai.tabular.data import TabularDataLoaders
from matplotlib import pyplot as plt
from pandas import DataFrame

from db import products, train, test
from utils import cut_date

WATER_ID = 5550022
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


def get_water(data_frame, geo_cluster_id):
    # test by water
    skus = get_sku_by_type(WATER_ID)
    water_in = data_frame.loc[data_frame['SKU'].isin(skus)]
    # group by geo
    water_in_geo = water_in.loc[water_in.geoCluster == geo_cluster_id].sort_values(
        'date')
    return water_in_geo


def group_by_weeks():
    water_in_geo = get_water(train.pd, GEO_CLUSTER)

    grouped_frame_1_week = cut_date(water_in_geo)
    grouped_frame_10_weeks = cut_date(
        water_in_geo,
        step_timedelta=timedelta(weeks=10)
    )
    return grouped_frame_1_week, grouped_frame_10_weeks


def remove_id_cluster(data_frame):
    data_frame = data_frame.drop(columns=['ID'])
    data_frame = data_frame.drop(columns=['geoCluster'])
    return data_frame


def test_learn():

    learn_frame: DataFrame = get_water(train.pd, GEO_CLUSTER)
    learn_frame = remove_id_cluster(learn_frame)
    learn_frame = learn_frame.loc[learn_frame.price.notnull()]

    print(learn_frame.corr())
    print(learn_frame)

    test_frame = get_water(test.pd, GEO_CLUSTER)
    test_frame = remove_id_cluster(test_frame)
    test_frame = test_frame.loc[test_frame.price.notnull()]
    test_frame['sales'] = 0

    dls = TabularDataLoaders.from_df(
        learn_frame,
        y_names='sales',
        cat_names=['SKU', 'date'],
        cont_names=['price', 'sales'],
        procs=[Categorify, FillMissing, Normalize]
    )

    print(type(dls), dls)
    #dls.show_batch()

    learn = tabular_learner(dls, metrics=accuracy)
    learn.lr_find()
    learn.fit_one_cycle(10)
    #learn.show_results()

    dl = learn.dls.test_dl(test_frame)
    print(learn.get_preds(dl=dl))


if __name__ == '__main__':
    # print_hi('PyCharm')
    test_learn()
