import numpy as np
import datetime as dt
from os import listdir


def find_min_year(mats):
    """
    Finds the earliest year present in collection of matrices.
    :param mats: list of matrices
    :return: min year
    """
    # Indices of date components in data matrix. First comes the year, then
    # the month and the day.
    date_indices = [1, 2, 3]
    min_date = dt.datetime.now().date()
    for mat in mats:
        # Iterate over all first dates and find earliest one.
        first_date = dt.date(*(mat[0, date_indices].astype(np.int32)))
        if first_date < min_date:
            min_date = first_date
    return min_date.year


def create_dates_mat(in_dir_path):
    """
    For each day present in data set find number of companies, which increased
    their price and overall number of companies recorded during that day.
    :param in_dir_path: path to directory with companies matrices
    :return: created 4-D matrix
    """
    # Load all matrices.
    mats = []
    for filename in listdir(in_dir_path):
        mat = np.load(in_dir_path + "/" + filename)
        mats.append(mat)
    # Values describing size of created matrix.
    min_year = find_min_year(mats)
    year_span = dt.datetime.now().date().year - min_year + 1
    num_months = 12
    num_days = 31
    values_per_day = 2
    print(min_year)
    dates_mat = np.zeros([year_span, num_months, num_days, values_per_day])
    # Indices of analyzed company matrix components.
    date_indices = [1, 2, 3]
    open_i = 4
    close_i = 5
    # Indices of values in dates matrix.
    all_count = 0
    profit_count = 1
    # Iterate over all matrices and over all of their rows and count them
    # in dates matrix.
    for mat in mats:
        for row in mat:
            date = dt.date(*(row[date_indices].astype(np.int32)))
            dates_mat[
                date.year - min_year, date.month - 1,
                date.day - 1, all_count
            ] += 1
            # If closing price was higher than opening add 1 to profit counter.#
            if row[close_i] > row[open_i]:
                dates_mat[
                    date.year - min_year, date.month - 1,
                    date.day - 1, profit_count
                ] += 1
    return dates_mat


def main():
    """
    Main function of the script.
    :return: -
    """
    in_dir_path = "../../data/averaged"
    dates_mat_path = "../../data/dates_matrix/dates_matrix"
    dates_mat = create_dates_mat(in_dir_path)
    np.save(dates_mat_path, dates_mat)
    start_date = dt.date(2000, 1, 1)
    end_date = dt.datetime.now().date()
    delta = dt.timedelta(days=1)
    current_date = start_date
    while current_date < end_date:

        current_date += delta


if __name__ == "__main__":
    main()