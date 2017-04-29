import numpy as np
import pickle


# Path to extracted data file.
EXTRACTED_DATA_FILE_PATH = "../../data/extracted/wiki_prices_data.npy"

# Range of smoothing for company data.
SMOOTHING_RANGE = 5

# Numeric columns of companies data matrix which will be extracted
# and index of column used for classification.
COLUMNS_TO_EXTRACT = [1, 2, 3, 4]
CLASSIFICATION_COLUMN = 4

# Length of time window present in one input vector for basic nn.
WINDOW_LENGTH = 100

# Constants for rise and decline classification.
NUMBER_OF_LABELS = 2
RISE = 1
DECLINE = 0

# Ids for first/second nn which have to be differentiated in data set creation.
FIRST_NN = 0
SECOND_NN = 1

# Sizes of test, validation and train set parts in fractions of whole set.
TRAIN = 0.70
VALIDATION = 0.15
TEST = 0.15


def smooth_numeric_data(data_matrix, columns_to_smooth, smoothing_range):
    """
    Smooth chosen numeric columns in data matrix along time axis
    (first dimension). Modify matrix in place.
    :param data_matrix: matrix of collected data
    :param columns_to_smooth: list of indices for columns which will be
                              smoothed
    :param smoothing_range: range of smoothing
    :return: -
    """

    # Copy matrix fragment, which will be smoothed and pad it with edge values.
    smoothed_data = data_matrix[:, columns_to_smooth]
    smoothed_data = np.pad(
        smoothed_data,
        ((smoothing_range // 2, smoothing_range // 2), (0, 0)),
        'edge'
    )

    # Smooth it.
    smoothed_data_copy = np.copy(smoothed_data)
    for i in range(smoothing_range // 2):
        smoothed_data += np.roll(smoothed_data_copy, i + 1, 0)
        smoothed_data += np.roll(smoothed_data_copy, -(i + 1), 0)
    smoothed_data = np.divide(smoothed_data, smoothing_range)

    # Replace old values in company data matrix with smoothed ones.
    data_matrix[:, columns_to_smooth] = smoothed_data[
        smoothing_range // 2:-(smoothing_range // 2), :
    ]


def split_data_matrix(data_matrix, window_length, columns_to_extract):
    """
    Splits data matrix for nn input vectors (time windows).
    :param data_matrix: matrix of data which will be split
    :param window_length: length of time window of input vector
    :param columns_to_extract: columns present in result input vectors
    :return: split matrix
    """

    # Derive number of available input vector from number of time points in
    # data matrix and window length.
    number_of_time_windows = data_matrix.shape[0] // window_length

    extracted_data = data_matrix[
        :number_of_time_windows * window_length, columns_to_extract
    ]
    split_matrix = np.reshape(
        extracted_data, [number_of_time_windows, -1]
    )

    return split_matrix


def get_data_set_part(
        company_data_matrix, destination,
        columns_to_extract, classification_column,
        window_length, smoothing_range
):
    """
    Turn company data matrix into fragment of first/second basic nn data set.
    :param company_data_matrix: matrix of collected company data
    :param destination: states whether data set will be used in
                                 first or second nn training
    :param columns_to_extract: indices of extracted from matrix columns
    :param classification_column: index of column used for classification
    :param window_length: length of time window (nn input vector)
    :param smoothing_range: performed in case of first nn destination
    :return: fragment of data set for this company (data and labels)
    """

    if destination == FIRST_NN:
        smooth_numeric_data(
            company_data_matrix, columns_to_extract, smoothing_range
        )

    split_company_matrix = split_data_matrix(
        company_data_matrix, window_length, columns_to_extract
    )

    # Find labels for each but last time window present in split matrix.
    # Last window is discarded, because there might not be enough data
    # to find label for it.
    labels = np.zeros([split_company_matrix.shape[0] - 1, NUMBER_OF_LABELS])
    values_per_time_point = len(columns_to_extract)
    classification_column_index = columns_to_extract.index(
        classification_column
    )
    for i in range(labels.shape[0]):
        window_last_value = split_company_matrix[
            i, -values_per_time_point + classification_column_index - 1
        ]

        if destination == FIRST_NN:
            next_value_index = values_per_time_point * \
                               ((smoothing_range // 2) * 2) + \
                               classification_column_index - 1
        else:
            next_value_index = classification_column_index - 1

        next_value = split_company_matrix[i + 1, next_value_index]
        if next_value > window_last_value:
            labels[i, RISE] = 1.0
        else:
            labels[i, DECLINE] = 1.0

    return split_company_matrix[:-1], labels


def create_data_set(
        company_data_matrix, destination,
        columns_to_extract, classification_column,
        window_length, smoothing_range
):
    """
    Transform data matrix into data set with labels prepared for first or
    second nn.
    :param data_matrix: data matrix to be turned into nn data set
    :param destination: describes destination of created data set (first
                        or second nn
    :param columns_to_extract: indices of extracted from matrix columns
    :param classification_column: index of column used for classification
    :param window_length: length of time window (nn input vector)
    :param smoothing_range: performed in case of first nn destination
    :return: created data set and its labels
    """

    data_parts = []
    labels_parts = []
    company_rows_indices = [0]
    for i in range(1, data_matrix.shape[0]):
        if data_matrix[i, 0] == data_matrix[company_rows_indices[-1], 0]:
            company_rows_indices.append(i)
        else:
            if len(company_rows_indices) >= 2 * window_length:
                data_part, labels_part = get_data_set_part(
                    data_matrix[company_rows_indices], destination,
                    columns_to_extract, classification_column,
                    window_length, smoothing_range
                )
                data_parts.append(data_part)
                labels_parts.append(labels_part)
            company_rows_indices = [i]

    data = np.concatenate(data_parts)
    labels = np.concatenate(labels_parts)

    # Determine split points.
    num_of_time_windows = data.shape[0]
    indices = np.random.permutation(num_of_time_windows)
    split_points = [
        int(TRAIN * num_of_time_windows),
        int((TRAIN + VALIDATION) * num_of_time_windows)
    ]
    train_indices, validation_indices, test_indices = np.split(
        indices,
        split_points
    )

    train_data = data[train_indices]
    train_labels = labels[train_indices]
    validation_data = data[validation_indices]
    validation_labels = labels[validation_indices]
    test_data = data[test_indices]
    test_labels = labels[test_indices]

    return train_data, train_labels, validation_data, \
           validation_labels, test_data, test_labels


if __name__ == "__main__":
    data_matrix = np.load(EXTRACTED_DATA_FILE_PATH)
    train_data, train_labels, validation_data, \
    validation_labels, test_data, test_labels = create_data_set(
        data_matrix, SECOND_NN, COLUMNS_TO_EXTRACT,
        CLASSIFICATION_COLUMN, WINDOW_LENGTH, SMOOTHING_RANGE
    )

    np.save("../../data/set/first_nn/train_data", train_data)
    np.save("../../data/set/first_nn/train_labels", train_labels)
    np.save("../../data/set/first_nn/validation_data", validation_data)
    np.save("../../data/set/first_nn/validation_labels", validation_labels)
    np.save("../../data/set/first_nn/test_data", test_data)
    np.save("../../data/set/first_nn/test_labels", test_labels)
