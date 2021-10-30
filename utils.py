from datetime import timedelta


def cut_date(pd, step_timedelta=timedelta(weeks=1), start_date=None, end_date=None):
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
    while start_date <=end_date:
        cut_date = start_date+step_timedelta
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