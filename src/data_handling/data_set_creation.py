import numpy as np


# Path to extracted data file and pattern for output directory.
EXTRACTED_DATA_FILE_PATH = "../../data/extracted/wiki_prices_data.npy"
OUT_DIRECTORY_PATTERN = "../../data/set/basic_nn_set_{}/"

# Range of smoothing for company data.
SMOOTHING_RANGE = 5

# Numeric columns of companies data matrix which will be extracted, smoothed
# and index of column used for classification.
COLUMNS_TO_EXTRACT = [1, 2, 3, 4, 5]
COLUMNS_TO_SMOOTH = [1, 2, 3, 4]
CLASSIFICATION_COLUMN = 4

# Length of time window present in one input vector for basic nn.
WINDOW_LENGTH = 30

# Constants for rise and decline classification.
NUMBER_OF_LABELS = 2
RISE = 1
DECLINE = 0

# Ids for created data sets and current id.
FIRST_SET = 0
SECOND_SET = 1
THIRD_SET = 2
CURRENT_SET = FIRST_SET

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

    :param data_matrix: data matrix which will be split
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
    split_matrix = extracted_data.reshape([number_of_time_windows, -1])

    return split_matrix


def normalize_time_window(time_window, values_per_time_point):
    """
    Bring all values in time window to -1, 1 range and center them around 0.

    :param time_window: vector of values for set of time points
    :param values_per_time_point: number of numeric values describing one time
                                  point
    :return: normalized time window vector
    """

    for i in range(values_per_time_point):

        # Subtract min numeric value from all values of the same kind present
        # in the time window.
        min_value = np.min(time_window[i::values_per_time_point])
        time_window[i::values_per_time_point] -= min_value

        # Divide by new max value to bring transformed parameters to range 0, 1
        max_value = np.max(time_window[i::values_per_time_point])
        time_window[i::values_per_time_point] /= max_value if max_value != 0.0 \
                                                 else 1.0

        # Subtract mean from all values to center them around 0.
        time_window[i::values_per_time_point] -= np.mean(
            time_window[i::values_per_time_point]
        )

    return time_window


def find_time_window_label(
        destination, time_window, next_time_window, smoothing_range,
        values_per_time_point, classification_column_index
):
    """
    Determine label for given time window basing on the next one.

    :param destination: id of set for which label is created
    :param time_window: analyzed vector of values
    :param next_time_window: time window which comes immediately after
                             analysed one
    :param smoothing_range: range of smoothing
    :param values_per_time_point: number of numeric values describing one time
                                  point
    :param classification_column_index: index of numeric value which will
                                        be base for label determination
    :return: created label which matches given time window
    """

    # Find last value of analysed time window.
    window_last_value = time_window[
        -values_per_time_point + classification_column_index
    ]

    # Find value in next time window which will determine the label.
    if destination == FIRST_SET:
        next_value_index = values_per_time_point * \
                           ((smoothing_range // 2) * 2) + \
                           classification_column_index
        next_value = next_time_window[next_value_index]
    elif destination == SECOND_SET:
        next_value_indices = [
            i * values_per_time_point + classification_column_index
            for i in range(smoothing_range)
        ]
        next_value = np.mean(next_time_window[next_value_indices])
    else:
        next_value_index = classification_column_index
        next_value = next_time_window[next_value_index]

    # Create label.
    label = [0.0] * NUMBER_OF_LABELS
    if next_value > window_last_value:
        label[RISE] = 1.0
    else:
        label[DECLINE] = 1.0

    return label


def get_data_set_part(
        company_data_matrix, destination,
        columns_to_extract, columns_to_smooth,
        classification_column, window_length, smoothing_range
):
    """
    Turn company data matrix into fragment of first basic nn data set.

    :param company_data_matrix: matrix of collected company data
    :param destination: id of set for which this part is created
    :param columns_to_extract: indices of columns extracted from data matrix
    :param columns_to_smooth: indices of columns smoothed in data matrix
    :param classification_column: index of column used for classification
    :param window_length: length of time window (nn input vector)
    :param smoothing_range: range for smoothing operation
    :return: fragment of data set for this company (data and labels)
    """

    # Transform company data matrix.
    if destination == FIRST_SET:
        smooth_numeric_data(
            company_data_matrix, columns_to_smooth, smoothing_range
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
    for i in range(split_company_matrix.shape[0] - 1):
        labels[i] = find_time_window_label(
            destination, split_company_matrix[i], split_company_matrix[i + 1],
            smoothing_range, values_per_time_point, classification_column_index
        )

    # Normalize all created time windows.
    for i in range(split_company_matrix.shape[0]):
        split_company_matrix[i] = normalize_time_window(
            split_company_matrix[i], values_per_time_point
        )

    return split_company_matrix[:-1], labels


def create_data_set(
        company_data_matrix, destination,
        columns_to_extract, columns_to_smooth, classification_column,
        window_length, smoothing_range
):
    """
    Transform data matrix into data set with labels prepared for first or
    second nn.
    :param data_matrix: data matrix to be turned into nn data set
    :param destination: describes destination of created data set (first
                        or second nn
    :param columns_to_extract: indices of extracted from matrix columns
    :param columns_to_smooth: indices of columns smoothed in data matrix
    :param classification_column: index of column used for classification
    :param window_length: length of time window (nn input vector)
    :param smoothing_range: performed in case of first nn destination
    :return: created data set and its labels
    """

    # Process all data in data matrix individually for all companies.
    data_parts = []
    labels_parts = []
    company_rows_indices = [0]
    for i in range(1, data_matrix.shape[0]):

        # Find fragment of matrix which contains current company and then...
        if data_matrix[i, 0] == data_matrix[company_rows_indices[-1], 0]:
            company_rows_indices.append(i)
        else:

            # Process company data if
            if len(company_rows_indices) >= 2 * window_length:
                data_part, labels_part = get_data_set_part(
                    data_matrix[company_rows_indices], destination,
                    columns_to_extract, columns_to_smooth,
                    classification_column, window_length, smoothing_range
                )
                data_parts.append(data_part)
                labels_parts.append(labels_part)
            company_rows_indices = [i]

    data = np.concatenate(data_parts)
    labels = np.concatenate(labels_parts)

    # Create set of data set indices in random order, which will be used to
    # split data set randomly for train, validation and test parts.
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

    # Divide data set and labels for train, validation and test parts.
    train_data = data[train_indices]
    train_labels = labels[train_indices]
    validation_data = data[validation_indices]
    validation_labels = labels[validation_indices]
    test_data = data[test_indices]
    test_labels = labels[test_indices]

    return train_data, train_labels, validation_data, \
           validation_labels, test_data, test_labels


def create_output_paths(set_id):
    """
    Create set of paths needed to save data set.

    :param set_id: Id of currently created set.
    :return: train, validation and test data, and labels out paths
    """

    out_dir_path = OUT_DIRECTORY_PATTERN.format(set_id)

    # Create needed paths.
    train_data_path = out_dir_path + "train_data"
    train_labels_path = out_dir_path + "train_labels"
    validation_data_path = out_dir_path + "validation_data"
    validation_labels_path = out_dir_path + "validation_labels"
    test_data_path = out_dir_path + "test_data"
    test_labels_path = out_dir_path + "test_labels"

    return train_data_path, train_labels_path, \
           validation_data_path, validation_labels_path, \
           test_data_path, test_labels_path


if __name__ == "__main__":

    # Load extracted data matrix and transform it.
    data_matrix = np.load(EXTRACTED_DATA_FILE_PATH)
    train_data, train_labels, validation_data, \
    validation_labels, test_data, test_labels = create_data_set(
        data_matrix, CURRENT_SET, COLUMNS_TO_EXTRACT, COLUMNS_TO_SMOOTH,
        CLASSIFICATION_COLUMN, WINDOW_LENGTH, SMOOTHING_RANGE
    )

    # Get save paths and save created matrices.
    train_data_path, train_labels_path, \
    validation_data_path, validation_labels_path, \
    test_data_path, test_labels_path = create_output_paths(CURRENT_SET)
    np.save(train_data_path, train_data)
    np.save(train_labels_path, train_labels)
    np.save(validation_data_path, validation_data)
    np.save(validation_labels_path, validation_labels)
    np.save(test_data_path, test_data)
    np.save(test_labels_path, test_labels)

    print(train_data.shape)
