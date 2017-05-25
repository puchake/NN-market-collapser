import numpy as np
import csv
import datetime as dt


def split_csv(in_file_path, out_dir_path):
    """
    Splits csv data file into smaller pieces using ticker names. It saves
    split data as numpy arrays.
    :param in_file_path: path to csv file
    :param out_dir_path: path to output directory
    :return: -
    """
    # Indices of extracted columns.
    ticker_i = 0
    date_i = 1
    open_i = 2
    close_i = 5
    volume_i = 6
    # Format of the dates contained in the csv file.
    date_format = "%Y-%m-%d"
    in_file = open(in_file_path)
    csv_reader = csv.reader(in_file)
    # Skip the header row.
    next(csv_reader)
    # Int id of the currently processed ticker. His name will be switched to
    # this id in the saved numpy array.
    current_ticker_id = 0
    previous_ticker = ""
    ticker_data = []
    for row in csv_reader:
        # Convert numeric values in the row. We have to take care of possible
        # empty strings.
        open_price = float(row[open_i] or 0.0)
        close_price = float(row[close_i] or 0.0)
        volume = float(row[volume_i] or 0.0)
        # Convert date to a tuple of (year, month, day)
        date = dt.datetime.strptime(row[date_i], date_format).date()
        date = date.year, date.month, date.day
        # Get the ticker name.
        ticker = row[ticker_i]
        if ticker != previous_ticker and previous_ticker != "":
            # Save accumulated data to numpy matrix file.
            ticker_mat = np.array(ticker_data)
            np.save(out_dir_path + "/" + previous_ticker, ticker_mat)
            ticker_data = []
            current_ticker_id += 1
        # Accumulate current ticker data.
        ticker_data.append(
            [current_ticker_id, *date, open_price, close_price, volume]
        )
        previous_ticker = ticker
    # Save the last ticker data.
    ticker_mat = np.array(ticker_data)
    np.save(out_dir_path + "/" + previous_ticker, ticker_mat)
    in_file.close()


def main():
    """
    Script main function.
    :return: -
    """
    # Parameters for split_csv call.
    # Paths.
    in_file_path = "../../data/raw/wiki_prices_data.csv"
    out_dir_path = "../../data/split"
    split_csv(in_file_path, out_dir_path)
    return


if __name__ == "__main__":
    main()