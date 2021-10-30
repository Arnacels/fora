import pandas
from pandas import DataFrame


class Base:
    file_name: str

    def __init__(self):
        self.pd: DataFrame = pandas.read_csv(self.file_name)

    def filter_by_sku(self, sku):
        return self.pd.loc[self.pd.SKU == sku]


class Train(Base):
    file_name = 'train_final.csv'


class Test(Base):
    file_name = 'test_data.csv'


class SKU(Base):
    file_name = 'sku_final.csv'

    def filter_by_type(self, commodity_group: str):
        return self.pd.loc[self.pd.commodity_group == commodity_group]


class GEO(Base):
    file_name = 'geo_params.csv'


geo = GEO()
products = SKU()
test = Test()
train = Train()
