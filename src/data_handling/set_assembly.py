import numpy as np
import datetime as dt
from os import listdir, path


def gather_mats(
        split_mat, avg_5_mat, avg_25_mat, avg_50_mat, dates_mat, min_year
):
    """
    Collects chosen columns from split and avg matrices and adds dates_mat
    indicator data for each row (each day).
    :param split_mat: original company data matrix
    :param avg_5_mat: matrix with EMA of length 5 of closing prices
    :param avg_25_mat: matrix with EMA of length 25 of closing prices
    :param avg_50_mat: matrix with EMA of length 50 of closing prices
    :param dates_mat: matrix of profit indicators for each date
    :return: matrix of gathered data
    """
    # Gather matrix columns indices.
    gather_split_i = 0
    gather_avg_5_i = 1
    gather_avg_25_i = 2
    gather_avg_50_i = 3
    gather_volume_i = 4
    gather_dates_indicator_i = 5
    # Indices of date fragment columns in split matrix.
    dates_indices = [1, 2, 3]
    # Indices of elements in dates matrix.
    all_i = 0
    profit_i = 1
    # Index of close price column and volume column.
    close_i = 5
    volume_i = 6
    # Number of gathered values. Original close price + 3 averages profit
    # indicator and volume will be collected.
    gathered_row_len = 6
    # Create gathered mat with row count of avg_50_mat as it is the shortest
    # of all input matrices.
    gathered_mat = np.zeros([avg_50_mat.shape[0], gathered_row_len])
    for i in range(avg_50_mat.shape[0]):
        # Gather split, avg_5, avg_25, avg_50 and volume columns.
        gathered_mat[-(i + 1), gather_split_i] = split_mat[-(i + 1), close_i]
        gathered_mat[-(i + 1), gather_avg_5_i] = avg_5_mat[-(i + 1), close_i]
        gathered_mat[-(i + 1), gather_avg_25_i] = avg_25_mat[-(i + 1), close_i]
        gathered_mat[-(i + 1), gather_avg_50_i] = avg_50_mat[-(i + 1), close_i]
        gathered_mat[-(i + 1), gather_volume_i] = split_mat[-(i + 1), volume_i]
        # Construct the date of current row and access dates matrix indicator.
        date = dt.date(*(split_mat[-(i + 1), dates_indices].astype(np.int32)))
        all_count = dates_mat[
            date.year - min_year, date.month - 1,
            date.day - 1, all_i
        ]
        profit_count = dates_mat[
            date.year - min_year, date.month - 1,
            date.day - 1, profit_i
        ]
        # Set indicator column element of current row to calculated indicator.
        gathered_mat[-(i + 1), gather_dates_indicator_i] = profit_count / \
                                                           all_count
    return gathered_mat


def label_mat(mat):
    """
    Assign labels to each row of gathered matrix.
    :param mat: previously gathered matrix
    :return: labels for gathered matrix rows
    """
    # Index and range of average used for labeling.
    gather_avg_25_i = 2
    avg_range = 25
    # Labels for rising and falling price.
    rising_i = 1
    falling_i = 0
    num_classes = 2
    labels = np.zeros([mat.shape[0] - avg_range + 1, num_classes])
    for i in range(mat.shape[0] - avg_range + 1):
        # If average 25 day price rises after 24 days assign rising label, else
        # assign falling label.
        if mat[i, gather_avg_25_i] < mat[i + avg_range - 1, gather_avg_25_i]:
            labels[i, rising_i] = 1.0
        else:
            labels[i, falling_i] = 1.0
    return labels


def normalize_mat(mat):
    """
    Bring all values in matrix to around -1, 1 range with mean 0.
    :param mat: matrix of gathered data
    :return: normalized matrix
    """
    # Gather matrix columns indices.
    gather_split_i = 0
    gather_avg_5_i = 1
    gather_avg_25_i = 2
    gather_avg_50_i = 3
    gather_volume_i = 4
    gather_dates_indicator_i = 5
    # Normalize prices. We want to keep relationship between prices
    # (eg. avg_5 > split) untouched, so we use single set of max and mean for
    # split and all averages.
    prices_indices = [
        gather_split_i, gather_avg_5_i, gather_avg_25_i, gather_avg_50_i
    ]
    mat[:, prices_indices] /= np.max(mat[:, prices_indices])
    mat[:, prices_indices] *= 2
    mat[:, prices_indices] -= np.mean(mat[:, prices_indices])
    # Normalize volume.
    mat[:, gather_volume_i] /= np.max(mat[:, gather_volume_i])
    mat[:, gather_volume_i] *= 2
    mat[:, gather_volume_i] -= np.mean(mat[:, gather_volume_i])
    # Subtract 1.0 from dates indicator multiplied by 2.0 as it is already in
    # range 0.0, 1.0 and we don't want characteristic values to vary between
    # matrices as it is data outside of one company scope.
    dates_indicator_mean = 1.0
    mat[:, gather_dates_indicator_i] *= 2
    mat[:, gather_dates_indicator_i] -= dates_indicator_mean
    return mat


def assemble_set(
        split_in_dir_path, avg_5_in_dir_path, avg_25_in_dir_path,
        avg_50_in_dir_path, dates_mat_path, min_year,
        data_out_dir_path, labels_out_dir_path
):
    """
    Gathers companies data, labels and normalizes it.
    :param split_in_dir_path: path to dir containing split matrices
    :param avg_5_in_dir_path: path to avg_5 matrices in dir
    :param avg_25_in_dir_path: path to avg_25 matrices in dir
    :param avg_50_in_dir_path: path to avg_50 matrices in dir
    :param dates_mat_path: path to dates matrix
    :param min_year: min year contained in companies data
    :param data_out_dir_path: path to data output dir
    :param labels_out_dir_path: path to labels output dir
    :return: -
    """
    # Minimal size of the gathered matrix.
    labeling_range = 25
    # Load dates matrix.
    dates_mat = np.load(dates_mat_path)
    for filename in listdir(split_in_dir_path):
        # If company matrix exists in all variants.
        if path.isfile(avg_5_in_dir_path + "/" + filename) and \
            path.isfile(avg_25_in_dir_path + "/" + filename) and \
            path.isfile(avg_50_in_dir_path + "/" + filename):
            # Load all matrices.
            split_mat = np.load(split_in_dir_path + "/" + filename)
            avg_5_mat = np.load(avg_5_in_dir_path + "/" + filename)
            avg_25_mat = np.load(avg_25_in_dir_path + "/" + filename)
            avg_50_mat = np.load(avg_50_in_dir_path + "/" + filename)
            # Gather data from them, label it and normalize if we have
            # enough data to label it.
            if avg_50_mat.shape[0] >= labeling_range:
                gathered_mat = gather_mats(
                    split_mat, avg_5_mat, avg_25_mat,
                    avg_50_mat, dates_mat, min_year
                )
                labels = label_mat(gathered_mat)
                labeled_rows = labels.shape[0]
                normalized_mat = normalize_mat(gathered_mat[:labeled_rows])
                # Save results.
                np.save(data_out_dir_path + "/" + filename, normalized_mat)
                np.save(labels_out_dir_path + "/" + filename, labels)


def main():
    """
    Main function of this script.
    :return: -
    """
    # Path used in assembly and previously discovered min year value.
    split_in_dir_path = "../../data/split"
    avg_5_in_dir_path = "../../data/averaged_5"
    avg_25_in_dir_path = "../../data/averaged_25"
    avg_50_in_dir_path = "../../data/averaged_50"
    dates_mat_path = "../../data/dates_matrix/dates_matrix.npy"
    min_year = 1962
    data_out_dir_path = "../../data/rnn_set/data"
    labels_out_dir_path = "../../data/rnn_set/labels"
    assemble_set(
        split_in_dir_path, avg_5_in_dir_path, avg_25_in_dir_path,
        avg_50_in_dir_path, dates_mat_path, min_year,
        data_out_dir_path, labels_out_dir_path
    )


if __name__ == "__main__":
    main()