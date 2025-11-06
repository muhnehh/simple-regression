# India's Population Regression and Prediction

This project performs linear regression analysis on India's historical population data and makes predictions for future years.

## Overview

The project uses historical population data from 1960 to 2020 to build a linear regression model. The model is then used to predict India's population for the years 2025 through 2050.

## Features

- Linear regression model trained on historical population data (1960-2020)
- Population predictions for future years (2025-2050)
- Statistical metrics (R² score, RMSE)
- Visual representation of historical data, regression line, and predictions

## Requirements

- Python 3.6+
- numpy
- matplotlib
- scikit-learn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/muhnehh/simple-regression.git
cd simple-regression
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the regression analysis:
```bash
python india_population_regression.py
```

The script will:
1. Train a linear regression model on historical data
2. Display model parameters and performance metrics
3. Generate predictions for future years (2025-2050)
4. Create and save a visualization as `india_population_prediction.png`

## Output

The script outputs:
- Model equation and parameters
- R² score and RMSE metrics
- Population predictions for future years
- A visualization graph saved as PNG

## Data Source

Historical population data is sourced from World Bank and Census of India, covering years 1960 to 2020.

## License

This project is for educational purposes.
