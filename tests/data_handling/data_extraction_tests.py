import csv
import pickle
import unittest
import numpy as np

import data_handling.data_extraction as data_extraction


class BasicRnnDataExtractionTestCase(unittest.TestCase):

    def test_extracted_wiki_prices_data_number_of_rows(self):

        # Arrange
        csv_file = open(data_extraction.RAW_DATA_FILE_PATH)
        csv_reader = csv.reader(csv_file)
        data_matrix = np.load(data_extraction.EXTRACTED_DATA_FILE_PATH)

        # Skip header row.
        next(csv_reader)

        # Act
        csv_row_count = sum(1 for row in csv_reader)

        # Assert
        self.assertEqual(csv_row_count, data_matrix.shape[0])

    def test_extracted_wiki_prices_data_numeric_values(self):

        # Arrange
        csv_file = open(data_extraction.RAW_DATA_FILE_PATH)
        csv_reader = csv.reader(csv_file)
        data_matrix = np.load(data_extraction.EXTRACTED_DATA_FILE_PATH)
        results = np.empty([data_matrix.shape[0], ], dtype=np.bool)

        # Skip header row.
        next(csv_reader)

        # Act
        for row, i in zip(csv_reader, range(data_matrix.shape[0])):
            results[i] = True
            csv_row = tuple(row)
            for column_index, j in zip(
                data_extraction.COLUMNS_TO_EXTRACT,
                range(1, data_matrix.shape[1])
            ):
                csv_value = float(
                    csv_row[column_index] if csv_row[column_index] else 0.0
                )
                results[i] = results[i] and (csv_value == data_matrix[i, j])

        # Assert
        self.assertTrue(np.alltrue(results))

    def test_extracted_wiki_prices_data_tickers_ids(self):

        # Arrange
        csv_file = open(data_extraction.RAW_DATA_FILE_PATH)
        csv_reader = csv.reader(csv_file)
        data_matrix = np.load(data_extraction.EXTRACTED_DATA_FILE_PATH)
        companies_dict = pickle.load(
            open(data_extraction.COMPANIES_DICT_FILE_PATH, "rb")
        )
        results = np.empty([data_matrix.shape[0], ], dtype=np.bool)

        # Skip header row.
        next(csv_reader)

        # Act
        for row, i in zip(csv_reader, range(data_matrix.shape[0])):
            csv_row = tuple(row)
            results[i] = companies_dict[
                             csv_row[data_extraction.TICKER_COLUMN]
                         ] == data_matrix[i, 0]

        # Assert
        self.assertTrue(np.alltrue(results))


if __name__ == '__main__':
    unittest.main()
