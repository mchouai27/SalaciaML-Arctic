# Traditional Quality Control for Oceanographic Data

This Python script provides functions to perform traditional quality control on oceanographic data, including temperature, salinity, depth, and pressure measurements. It detects various types of outliers and flags them accordingly.

## Dependencies

Make sure you have the following libraries installed:

- pandas
- numpy
- sw (Seawater library)
- hampel (for outlier detection)

You can install them using pip:

```bash
pip install pandas numpy seawater hampel
```
## Usage

1. **Import necessary libraries**:

    ```python
    import pandas as pd
    import numpy as np
    from seawater.library import T90conv
    from seawater import eos80 as sw
    from hampel import hampel
    ```

2. **Load your data**:

    Ensure your data is in CSV format.

    ```python
    data_file = 'YourData.csv'
    data = pd.read_csv(data_file)
    ```

3. **Run traditional quality control**:

    Call the `check_data` function and pass your data as an argument. This function performs various quality control checks and returns processed data.

    ```python
    processed_data = check_data(data)
    ```

4. **Save processed data**:

    Save the processed data to a new CSV file.

    ```python
    processed_data.to_csv('processed_data.csv', index=False)
    ```

## Functions

- `check_data(data)`: Checks the integrity of data and performs quality control. Returns processed data.

- `Bottom_Top_Temp_Outliers(data)`: Detects bottom and top temperature outliers.

- `Traditional_outlier_detection(data)`: Detects traditional outliers in temperature.

- `T_Suspect_gradient_D_T(data)`: Calculates temperature suspect gradient based on depth.

- `T_Suspect_gradient_T_D(data)`: Calculates temperature suspect gradient based on temperature.

- `Small_Temp_Outliers_below_mixed_layer(data)`: Detects small temperature outliers below the mixed layer.

- `T_Suspect_gradient(data)`: Calculates temperature suspect gradient.

- `Miss_temperature_value(data)`: Detects missing temperature values.

- `density_inversion_detection(data)`: Detects density inversion.

## Example

```python
import pandas as pd
import numpy as np
from seawater.library import T90conv
from seawater import eos80 as sw
from hampel import hampel

# Define data file path (use relative path)
data_file = 'YourData.csv'

# Read your data
data = pd.read_csv(data_file)

# Perform quality control on data
processed_data = check_data(data)

# Save processed data to CSV
processed_data.to_csv('processed_data.csv', index=False)
```
Ensure that you replace 'YourData.csv' with the path to your actual data file. You can also adjust the parameters in the quality control functions as needed for your specific dataset.

