from arima_files.sarimax_handler import handle_sarimax
from arima_files.arima_handler import handle_arima, run_single
import sys

if __name__ == "__main__":
    if sys.argv[1] == "--type":
        if sys.argv[2] == "A":
            handle_arima()
        elif sys.argv[2] == "S":
            handle_sarimax()
        elif sys.argv[2] == "AS":
            run_single(0, 0, 2)