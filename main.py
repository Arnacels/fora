from datetime import timedelta

from pandas import DataFrame

from db import train, test
from filters import filter_by_sku_type_id, filter_by_cluster_id
from learner import LearnerCategoryGeo
from split_date import SplitDate
from utils import cut_date

WATER_ID = 5550022
TROPIC_ID = 5551018
YOGURT_ID = 5550259

GEO_CLUSTER = 2053
CITY_ID = 1


def group_by_weeks():
    water = filter_by_sku_type_id(train.pd, WATER_ID)
    water_in_geo = filter_by_cluster_id(water, GEO_CLUSTER)

    grouped_frame_1_week = cut_date(water_in_geo)
    grouped_frame_10_weeks = cut_date(
        water_in_geo,
        step_timedelta=timedelta(weeks=10)
    )
    return grouped_frame_1_week, grouped_frame_10_weeks


def test_learn():

    frame_skus_group_test = filter_by_cluster_id(test.pd, GEO_CLUSTER)
    frame_skus_group_test = filter_by_sku_type_id(frame_skus_group_test, TROPIC_ID)
    test_frame: DataFrame = frame_skus_group_test
    test_frame = test_frame.drop(columns=['ID'])
    test_frame = test_frame.drop(columns=['geoCluster'])
    test_frame = test_frame.loc[test_frame.price.notnull()]
    test_frame = SplitDate(test_frame).split()
    test_frame['sales'] = 0

    learner = LearnerCategoryGeo(
        train.pd,
        TROPIC_ID,
        GEO_CLUSTER
    )
    learn = learner.learn()
    # learn.show_results()

    # dl = learn.dls.test_dl(test_frame)
    # preds, targs = learn.get_preds(dl=dl)
    # print([i for i in preds.tolist()], targs)


if __name__ == '__main__':
    # print_hi('PyCharm')

    test_learn()
