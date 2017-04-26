import unittest
import numpy as np

import data_handling.data_set_creation as data_set_creation


class DataSetCreationTestCase(unittest.TestCase):

    def test_smooth_numeric_data(self):

        # Arrange
        matrix = np.arange(100).reshape([10, 10])
        columns_to_smooth = [1, 2, 3]
        smoothing_range = 5
        expected_matrix = matrix
        expected_matrix = np.pad(expected_matrix, ((2, 2), (0, 0)), 'edge')
        smoothed_fragment = expected_matrix[:, columns_to_smooth]
        smoothed_fragment_copy = np.copy(smoothed_fragment)
        for i in range(expected_matrix.shape[0] - smoothing_range // 2 - 1):
            smoothed_fragment[i + smoothing_range // 2] = \
                np.mean(smoothed_fragment_copy[i:i + smoothing_range], axis=0)
        expected_matrix[:, columns_to_smooth] = smoothed_fragment
        expected_matrix = expected_matrix[
            smoothing_range // 2:-(smoothing_range // 2)
        ]

        # Act
        data_set_creation.smooth_numeric_data(
            matrix, columns_to_smooth, smoothing_range
        )

        # Assert
        self.assertTrue(np.alltrue(matrix == expected_matrix))

    def test_split_data_matrix(self):

        # Arrange
        matrix = np.arange(100).reshape([10, 10])
        columns_to_extract = [1, 2, 3]
        window_length = 3
        expected_matrix = [[1, 2, 3, 11, 12, 13, 21, 22, 23],
                           [31, 32, 33, 41, 42, 43, 51, 52, 53],
                           [61, 62, 63, 71, 72, 73, 81, 82, 83]]

        # Act
        result = data_set_creation.split_data_matrix(
            matrix, window_length, columns_to_extract
        )

        # Assert
        self.assertTrue(np.alltrue(result == expected_matrix))


if __name__ == '__main__':
    unittest.main()
