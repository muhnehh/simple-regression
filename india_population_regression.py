"""
India's Population Regression and Prediction

This script performs linear regression on India's historical population data
and makes predictions for future years.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


# Historical population data for India (in millions)
# Source: World Bank and Census of India
years = np.array([1960, 1970, 1980, 1990, 2000, 2010, 2020])
population = np.array([450.5, 555.2, 698.9, 873.3, 1056.6, 1234.3, 1380.0])


def train_regression_model(X, y):
    """
    Train a linear regression model on the given data.
    
    Args:
        X: Independent variable (years)
        y: Dependent variable (population)
    
    Returns:
        Trained LinearRegression model
    """
    model = LinearRegression()
    model.fit(X, y)
    return model


def predict_population(model, future_years):
    """
    Predict population for future years using the trained model.
    
    Args:
        model: Trained regression model
        future_years: Array of years to predict
    
    Returns:
        Predicted population values
    """
    return model.predict(future_years)


def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        Tuple of (R² score, RMSE)
    """
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, rmse


def visualize_results(X_train, y_train, X_future, y_future_pred, model):
    """
    Create visualization of the regression results.
    
    Args:
        X_train: Training years
        y_train: Training population data
        X_future: Future years
        y_future_pred: Predicted future population
        model: Trained model
    """
    plt.figure(figsize=(12, 6))
    
    # Plot historical data
    plt.scatter(X_train, y_train, color='blue', s=100, alpha=0.6, label='Historical Data')
    
    # Plot regression line for historical period
    X_line = np.linspace(X_train.min(), X_future.max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    plt.plot(X_line, y_line, 'r--', linewidth=2, label='Regression Line')
    
    # Plot predictions
    plt.scatter(X_future, y_future_pred, color='green', s=100, alpha=0.6, 
                marker='^', label='Predicted Data')
    
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Population (millions)', fontsize=12)
    plt.title("India's Population Regression and Prediction", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('india_population_prediction.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'india_population_prediction.png'")
    plt.show()


def main():
    """
    Main function to run the regression analysis and prediction.
    """
    print("=" * 60)
    print("India's Population Regression and Prediction")
    print("=" * 60)
    print()
    
    # Reshape data for sklearn
    X = years.reshape(-1, 1)
    y = population
    
    # Train the model
    print("Training linear regression model...")
    model = train_regression_model(X, y)
    
    # Display model parameters
    print(f"\nModel Parameters:")
    print(f"  Coefficient (slope): {model.coef_[0]:.4f} million/year")
    print(f"  Intercept: {model.intercept_:.4f} million")
    print(f"  Equation: Population = {model.coef_[0]:.4f} × Year + {model.intercept_:.4f}")
    print()
    
    # Make predictions on training data
    y_pred_train = model.predict(X)
    
    # Calculate and display metrics
    r2, rmse = calculate_metrics(y, y_pred_train)
    print(f"Model Performance on Historical Data:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.2f} million")
    print()
    
    # Predict future population
    future_years = np.array([2025, 2030, 2035, 2040, 2045, 2050])
    X_future = future_years.reshape(-1, 1)
    y_future_pred = predict_population(model, X_future)
    
    # Display predictions
    print("Population Predictions:")
    print("-" * 40)
    for year, pop in zip(future_years, y_future_pred):
        print(f"  {year}: {pop:.2f} million ({pop/1000:.3f} billion)")
    print()
    
    # Visualize results
    print("Generating visualization...")
    visualize_results(X, y, X_future, y_future_pred, model)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
