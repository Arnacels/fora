from datetime import datetime


class SplitDate(object):

    def __init__(self, df):
        self._df = df
        self._dates = {
            'year': [],
            'month': [],
            'day': [],
            'weekday': []
        }

    def split(self):
        for index, series in self._df.iterrows():
            date = series.date
            self._split_date(date)
        for name_column, values in self._dates.items():
            self._df[name_column] = values
        self._df = self._df.drop(columns=['date'])
        return self._df

    def _split_date(self, date):
        date = datetime.strptime(date, '%Y-%m-%d')
        weekday = date.weekday()
        self._dates['year'].append(date.year)
        self._dates['month'].append(date.month)
        self._dates['day'].append(date.day)
        self._dates['weekday'].append(weekday)
