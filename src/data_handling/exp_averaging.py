import numpy as np
from matplotlib import pyplot as plt
from os import listdir


def exp_moving_average(vec, a):
    """
    Calculates EMA from given vector and alpha parameter.
    :param vec: input vector
    :param a: alpha parameter
    :return: calculated average
    """
    # Create elements multipliers vector. 1 for first element in vec and alpha
    # for every other element.
    multi_vec = np.ones(vec.shape)
    multi_vec[1:] = a
    exp_vec = np.flip(np.arange(vec.shape[0]), 0)
    avg = np.sum(np.multiply(multi_vec, np.multiply(vec, (1 - a) ** exp_vec)))
    return avg


def average_mat(mat, avg_range):
    """
    Average open and close prices in given matrix with avg range.
    :param mat: input matrix
    :param avg_range: range of EMA
    :return: new matrix with averaged open and close column
    """
    # Indices of open and close columns.
    open_i = 4
    close_i = 5
    avg_mat = np.zeros([mat.shape[0] - avg_range + 1, mat.shape[1]])
    for i in range(avg_mat.shape[0]):
        avg_mat[i] = mat[i + avg_range - 1]
        # Calculate averaged open and close prices.
        avg_mat[i, open_i] = exp_moving_average(
            mat[i:i + avg_range, open_i], 1 / avg_range
        )
        avg_mat[i, close_i] = exp_moving_average(
            mat[i:i + avg_range, close_i], 1 / avg_range
        )
    return avg_mat


def average_mats(in_dir_path, out_dir_path, avg_range):
    """
    Average all matrices present in in_dir_path with given range.
    :param in_dir_path: path to input dir
    :param out_dir_path: path to output dir
    :param avg_range: length of exp moving average
    :return:
    """
    # Average and save every company matrix.
    for filename in listdir(in_dir_path):
        mat = np.load(in_dir_path + "/" + filename)
        # Convert matrix only if it contains enough rows.
        if mat.shape[0] >= avg_range:
            avg_mat = average_mat(mat, avg_range)
            np.save(out_dir_path + "/" + filename, avg_mat)


def main():
    """
    Main function of this script.
    :return: -
    """
    # Parameters for averaging with length 5, 25 and 50.
    in_dir_path = "../../data/split"
    out_dir_path_1 = "../../data/averaged_50"
    out_dir_path_2 = "../../data/averaged_25"
    out_dir_path_3 = "../../data/averaged_5"
    avg_range_1 = 50
    avg_range_2 = 25
    avg_range_3 = 5
    average_mats(in_dir_path, out_dir_path_1, avg_range_1)
    average_mats(in_dir_path, out_dir_path_2, avg_range_2)
    average_mats(in_dir_path, out_dir_path_3, avg_range_3)


if __name__ == "__main__":
    main()