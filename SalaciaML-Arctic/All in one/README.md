# Data Processing and Quality Control

This Python script processes oceanographic data, performs quality control checks, and predicts data quality using a machine learning model. The script takes input data in CSV format, processes it, checks for various data quality issues, and provides predictions for problematic data points.

## Prerequisites

Before running the script, ensure you have the following installed:

- Python (>=3.6)
- Required Python libraries (Pandas, NumPy, Seawater, TensorFlow, Scikit-Learn)
- Trained machine learning model and scaler (saved as files)

## Usage

### Command-Line Arguments

- `--input`: Path to the input CSV file containing raw data.
- `--output`: Path to save the processed data with quality flags and machine learning predictions.
- `--model`: Path to the trained machine learning model file.
- `--scaler`: Path to the scaler used for preprocessing input features.

### Running the Script

```bash
python data_processing.py --input input_data.csv --output processed_data.csv --model model.h5 --scaler scaler.pkl
```

## Command-line Arguments

- **--input**: Path to the input CSV file containing oceanographic data.
- **--output**: Path to save the processed data with quality flags and predictions.
- **--model**: Path to the trained machine learning model file (in HDF5 format).
- **--scaler**: Path to the scaler file used for feature scaling (in Pickle format).

## Understanding `check_result` Flag

The script uses the `check_result` flag to provide information about potential issues with the input data. Here are the possible values and their meanings:

- **0**: Data is valid, and processing can proceed.
- **1**: Invalid syntax or non-float values present in the input data.
- **2**: Missing one or more required columns in the input data.
- **3**: At least one optional column is required.


## Output

The processed data will be saved in the specified output file (e.g., `processed_data.csv`). This file contains additional columns:

- **QF_trad**: Quality flag indicating issues detected using traditional methods.
- **ML_predictions**: Predictions from the machine learning model (1 for problematic data points, 0 for valid data points).

## Additional Notes

- If `check_result` is not 0, the script will not proceed with further processing.
- Ensure that the machine learning model and scaler files provided as input are compatible with the script.
- **Note:** The script suppresses TensorFlow info messages for a cleaner output.

## Dependencies

Make sure you have the following Python packages installed:

- `pandas`
- `numpy`
- `seawater`
- `hampel`
- `tensorflow`
- `scikit-learn`

## License

This project is licensed under the project M-VRE: The MOSAiC - Virtual Research Environment


