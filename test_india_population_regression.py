"""
Tests for India's Population Regression Model
"""

import numpy as np
from india_population_regression import (
    train_regression_model,
    predict_population,
    calculate_metrics
)


def test_regression_model():
    """Test that the regression model trains correctly."""
    # Sample data
    X = np.array([1960, 1970, 1980, 1990, 2000, 2010, 2020]).reshape(-1, 1)
    y = np.array([450.5, 555.2, 698.9, 873.3, 1056.6, 1234.3, 1380.0])
    
    # Train model
    model = train_regression_model(X, y)
    
    # Check that model has been trained
    assert model is not None
    assert hasattr(model, 'coef_')
    assert hasattr(model, 'intercept_')
    
    print("✓ Model training test passed")


def test_predictions():
    """Test that predictions are reasonable."""
    # Sample data
    X = np.array([1960, 1970, 1980, 1990, 2000, 2010, 2020]).reshape(-1, 1)
    y = np.array([450.5, 555.2, 698.9, 873.3, 1056.6, 1234.3, 1380.0])
    
    # Train model
    model = train_regression_model(X, y)
    
    # Make predictions
    future_years = np.array([2030, 2040, 2050]).reshape(-1, 1)
    predictions = predict_population(model, future_years)
    
    # Check that predictions are positive and increasing
    assert len(predictions) == 3
    assert all(predictions > 0)
    assert predictions[0] < predictions[1] < predictions[2]
    
    print("✓ Predictions test passed")


def test_metrics():
    """Test that metrics calculation works."""
    # Sample data
    y_true = np.array([100, 200, 300, 400, 500])
    y_pred = np.array([110, 190, 310, 390, 510])
    
    # Calculate metrics
    r2, rmse = calculate_metrics(y_true, y_pred)
    
    # Check that metrics are in valid ranges
    assert 0 <= r2 <= 1
    assert rmse >= 0
    
    print("✓ Metrics calculation test passed")


def test_model_performance():
    """Test that the model achieves good performance on historical data."""
    # Historical data
    X = np.array([1960, 1970, 1980, 1990, 2000, 2010, 2020]).reshape(-1, 1)
    y = np.array([450.5, 555.2, 698.9, 873.3, 1056.6, 1234.3, 1380.0])
    
    # Train and evaluate
    model = train_regression_model(X, y)
    y_pred = model.predict(X)
    r2, rmse = calculate_metrics(y, y_pred)
    
    # Model should have high R² (> 0.95) for linear trend
    assert r2 > 0.95, f"R² score {r2} is too low"
    
    # RMSE should be reasonable (< 50 million)
    assert rmse < 50, f"RMSE {rmse} is too high"
    
    print(f"✓ Model performance test passed (R²={r2:.4f}, RMSE={rmse:.2f})")


def run_all_tests():
    """Run all tests."""
    print("Running tests for India's Population Regression Model")
    print("=" * 60)
    
    try:
        test_regression_model()
        test_predictions()
        test_metrics()
        test_model_performance()
        
        print("=" * 60)
        print("All tests passed! ✓")
        return True
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
