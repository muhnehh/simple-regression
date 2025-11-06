# Population Regression for Future Prediction (India)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)]() [![License](https://img.shields.io/badge/License-MIT-green)]() [![Dataset](https://img.shields.io/badge/Dataset-World%20Bank%20SP.POP.TOTL-informational)]()

---

## 📌 Executive Summary

This project builds an **interpretable linear regression model** to forecast **India's total population** using **World Bank** data from **1960–2023**. The model is trained with **least squares** and used to predict **2025–2030** values. Outputs include a **regression plot**, a **predictions table**, and **growth-rate estimates**.

- ✅ **Simple & transparent**: classic linear regression with clear parameters  
- 📈 **Solid baseline**: performs well when long-term trends are approximately linear  
- 🧮 **Calculus link**: slope/intercept interpretation, rate-of-change (growth)  
- 🧪 **Reproducible**: precise data cleaning & deterministic pipeline

> **Key finding:** Forecasts suggest **steady growth** with a **gradually slowing rate** (~1.1% → ~1.10% by 2030), consistent with demographic transition expectations.

---

## 📂 Repository Structure

```
population-regression-india/
├─ README.md                  # This file
├─ data/
│  └─ API_SP.POP.TOTL_DS2_en_csv_v2.csv   # World Bank CSV (place here)
├─ src/
│  └─ population_regression.py            # (Optional) script version of notebook
├─ notebooks/
│  └─ analysis.ipynb                      # EDA + modeling + plots
├─ outputs/
│  ├─ population_regression_plot.png      # Saved figure
│  └─ predictions_2025_2030.csv           # Forecast table
└─ requirements.txt
```

> You can work entirely from a **notebook**, or use the **script** form in `src/` with the same logic.

---

## 🧪 Data

- **Source:** World Bank Open Data — *Total Population* (`SP.POP.TOTL`) for **India**  
- **Period:** 1960–2023  
- **Link (reference):** https://data.worldbank.org/indicator/SP.POP.TOTL?locations=IN

**Cleaning steps:**
1. Read CSV (`skiprows=4`) and filter `Country Name == 'India'`
2. Drop non-year metadata columns
3. Transpose to long format → columns: `Year`, `Population`
4. Convert to numeric types and drop missing values

---

## 🔧 Setup

### Option A — Quickstart (Notebook)

```bash
# Create environment
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install --upgrade pip wheel
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
# Open notebooks/analysis.ipynb
```

### Option B — Script

```bash
python src/population_regression.py   --csv data/API_SP.POP.TOTL_DS2_en_csv_v2.csv   --country "India"   --out-plot outputs/population_regression_plot.png   --out-csv outputs/predictions_2025_2030.csv
```

**Minimal `requirements.txt`:**
```
pandas>=1.5
numpy>=1.21
matplotlib>=3.5
scikit-learn>=1.1
jupyter>=1.0
```

---

## 📘 Methodology

We fit a **simple linear regression** model:
\[
\textsf{Population} = a \cdot \textsf{Year} + b
\]
where **\(a\)** is the **slope** (average annual change) and **\(b\)** the **intercept**.

- Estimation via **ordinary least squares (OLS)**
- Training data: **1960–2023**
- Forecast horizon: **2025–2030**
- **Growth rate** computed from year-over-year percentage change

**Code skeleton (matching the project’s logic):**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1) Load & clean
df = pd.read_csv("data/API_SP.POP.TOTL_DS2_en_csv_v2.csv", skiprows=4)
india_df = df[df['Country Name'] == 'India']
india_df = india_df.drop(columns=['Country Code', 'Indicator Name', 'Indicator Code'])
india_df = india_df.set_index('Country Name').T.reset_index()
india_df.columns = ['Year', 'Population']
india_df = india_df[~india_df['Year'].str.contains("Unnamed", na=False)]
india_df['Year'] = india_df['Year'].astype(int)
india_df['Population'] = pd.to_numeric(india_df['Population'], errors='coerce')
india_df = india_df.dropna()

# 2) Train
X = india_df[['Year']]
y = india_df['Population']
model = LinearRegression().fit(X, y)

# 3) Predict 2025–2030
future_years = pd.DataFrame({'Year': np.arange(2025, 2031)})
future_predictions = model.predict(future_years)

# 4) Plot
plt.figure(figsize=(9,5))
plt.scatter(india_df['Year'], india_df['Population'], s=12, label="Actual")
plt.plot(india_df['Year'], model.predict(X), label="Fit (OLS)")
plt.plot(future_years['Year'], future_predictions, linestyle='--', label="Forecast")
plt.title("India Population: Regression & Forecast (1960–2030)")
plt.xlabel("Year"); plt.ylabel("Population")
plt.legend(); plt.tight_layout()
plt.savefig("outputs/population_regression_plot.png", dpi=300)

# 5) Growth rates
future_df = future_years.copy()
future_df['Population'] = future_predictions.astype(int)
future_df['Growth Rate (%)'] = future_df['Population'].pct_change() * 100
future_df.to_csv("outputs/predictions_2025_2030.csv", index=False)
```

---

## 📊 Results

### Figure 1 — India Population Regression & Prediction
Saved to: `outputs/population_regression_plot.png`

<img width="825" height="495" alt="image" src="https://github.com/user-attachments/assets/af0566e6-755f-4abf-ab78-e20744369b6f" />

<img width="761" height="570" alt="image" src="https://github.com/user-attachments/assets/f227806b-dcf2-4ec0-a9d2-42a04ffa35e5" />


### Table — Predicted Values (2025–2030)

| Year | Predicted Population | Growth Rate (%) |
|-----:|---------------------:|----------------:|
| 2025 | 1,481,959,263 | — |
| 2026 | 1,498,971,328 | 1.15 |
| 2027 | 1,515,983,394 | 1.14 |
| 2028 | 1,532,995,459 | 1.12 |
| 2029 | 1,550,007,525 | 1.11 |
| 2030 | 1,567,019,591 | 1.10 |

> **Interpretation:** A steady upward trend with a **slight deceleration** in annual growth rate, aligning with demographic transition theory for maturing economies.

---

## 🔬 EDA Summary

- Time series from **1960–2023** shows a near-linear upward trend.  
- No missing values after India-only filtering and numeric coercion.  
- Linear model is appropriate as a **baseline**; residuals are small and trend-consistent.

---

## 🧭 Discussion & Limitations

- Linear regression **does not** capture saturation effects or nonlinearities.
- Forecasts assume **trend continuation** with no structural breaks (e.g., policy, pandemics).
- For medium/long-term planning, consider:  
  - **Logistic growth** (capacity limits), **polynomial regression**  
  - **Time-series models**: ARIMA/SARIMA  
  - **Feature-rich models**: fertility, mortality, urbanization, migration

---

## 🗺️ Roadmap (Future Work)

- [ ] Logistic or Gompertz growth model comparison  
- [ ] ARIMA baseline with AIC/BIC model selection  
- [ ] Multivariate regression including fertility/mortality/urbanization  
- [ ] Prediction intervals (95% CI) and error bars  
- [ ] Interactive dashboard (Streamlit) for scenario testing

---

## 📚 References

1. World Bank Open Data — *Total Population (SP.POP.TOTL), India*  
   https://data.worldbank.org/indicator/SP.POP.TOTL?locations=IN  
2. Scikit-learn — *LinearRegression*  
   https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

---

## 📝 License

This repository is released under the **MIT License**. See `LICENSE` for details.

---

## 🙌 Acknowledgments

- Department of Mathematics & Sciences, *Calculus for IT*  
- World Bank Open Data portal  
- Scikit-learn & the open-source Python ecosystem

Nehan, M. (2025). Population Regression for Future Prediction (India). Department of Mathematics & Sciences, Calculus for IT.
```
