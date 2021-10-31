from fastai.data.transforms import RandomSplitter
from fastai.metrics import rmse
from fastai.tabular.core import TabularPandas, Categorify, FillMissing
from fastai.tabular.learner import tabular_learner
from fastcore.basics import range_of
from pandas import DataFrame
from sklearn.metrics import r2_score

from filters import filter_by_sku_type_id
from split_date import SplitDate
from utils import set_mean_instead_of_null


class AbstractLearner(object):
    NAME: str

    CONT_NAMES: list
    CAT_NAMES: list
    Y_NAMES: str
    PREPROCESSORS: list
    BS_CHUNK: int = 2048

    def __init__(self, df):
        self._df: DataFrame = df
        self._dls = None

    def learn(self):
        raise NotImplementedError

    def _prepare_data(self):
        return

    def _make_loader(self):
        splits = self._make_splits()
        to = TabularPandas(
            self._df,
            y_names=self.Y_NAMES,
            cat_names=self.CAT_NAMES,
            cont_names=self.CONT_NAMES,
            procs=self.PREPROCESSORS,
            splits=splits
        )
        _dls = to.dataloaders(2048)
        return _dls

    def _make_learner(self, dls):
        self._learn = tabular_learner(dls)
        return self._learn

    def _save_model(self):
        self._learn.save()

    def _make_splits(self):
        self._splits = RandomSplitter(valid_pct=0.2)(range_of(self._df))
        return self._splits


class LearnerCategoryGeo(AbstractLearner):
    GROUP_COLUMN: str = 'SKU'
    CONT_NAMES: list = ['price', 'sales']
    CAT_NAMES: list = [
        #'geoCluster',
        'SKU',
        'year',
        'month',
        'day',
        'weekday']
    Y_NAMES: str = 'sales'
    PREPROCESSORS: list = [Categorify, FillMissing]

    def __init__(self, df, category_id, geo_id):
        super().__init__(df)
        self.category_id = category_id
        self.geo_id = geo_id

    def learn(self):
        self._prepare_data()
        self._dls = self._make_loader()

        self._dls.show_batch()

        self._learn = self._make_learner(self._dls)
        self._learn.lr_find(start_lr=1e-05, end_lr=1e+05, num_it=100 )
        self._learn.fit_one_cycle(150)
        return self._learn

    def _prepare_data(self):
        self._df = filter_by_sku_type_id(self._df, self.category_id)
        self._df = set_mean_instead_of_null(
            self._df,
            self.GROUP_COLUMN,
            self.CONT_NAMES
        )
        self._df = self.__clean_columns()
        self._df = SplitDate(self._df).split()
        print(self._df.corr(), self._df)

    def _make_learner(self, dls):
        self._learn = tabular_learner(
            dls,
            layers=[300, 200, 100, 50],
            model_dir='models',
            metrics=[rmse, r2_score]
        )
        return self._learn

    def __clean_columns(self):
        self._df = self._df.drop(columns=['ID'])
        self._df = self._df.drop(columns=['geoCluster'])
        return self._df
