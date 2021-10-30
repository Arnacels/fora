

class ValidateColumns(object):
    CAT_COLUMNS = ['price, sales']

    def __init__(self, df):
        self._df = df
        self._corr = None

    def validate(self):
        self._get_corr()
        self.removed_columns = []
        for index, series in self._corr.iterrows():
            if str(index) in self.CAT_COLUMNS:
                continue
            if series.sales < 0:
                self.removed_columns.append(index)
        self._df = self._df.drop(columns=self.removed_columns)
        return self._df

    def get_cat_columns(self):
        columns = list(self._df.columns)
        print(columns)
        for column in self.CAT_COLUMNS:
            if column in columns:
                columns.remove(column)
        return columns

    def _get_corr(self):
        self._corr = self._df.corr()
