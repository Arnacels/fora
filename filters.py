from utils import get_sku_by_type, get_geo_clusters_by_city


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