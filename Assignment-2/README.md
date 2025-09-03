# Assignment 2: Bias and Fairness Analysis in Absenteeism Prediction

## Overview

This project presents a comprehensive analysis of bias and fairness in the Absenteeism at Work dataset. We implemented a linear regression model to predict absenteeism hours and conducted thorough bias evaluation, corrective measures, and fairness assessment.

## Team Members

- **Krishna Kumar Bais** (241110038)
- **Rohan** (241110057)

## Dataset Description

The Absenteeism at Work dataset contains:
- **Total Records:** 740 observations
- **Features:** 21 attributes including demographic, behavioral, and workplace factors
- **Target Variable:** Absenteeism time in hours (range: 0-120 hours, mean: 6.92 hours)
- **Data Quality:** No missing values, 34 duplicate records identified
- **Employees:** 36 unique employees across multiple time periods

### Key Features
- **Demographic:** Age, Education, Service time
- **Behavioral:** Social drinker, Social smoker, Pet ownership
- **Workplace:** Transportation expense, Distance to work, Work load, Hit target
- **Health:** Weight, Height, Body mass index
- **Temporal:** Month of absence, Day of the week, Seasons

## Project Structure

```
Fair-Absenteeism-Analysis/
├── README.md                        # This file
├── Assignment-2.pdf                 # Assignment instructions
├── absenteeism.py                   # Main analysis script
├── Report.pdf                       # Report
└── Absenteeism_at_work.csv          # Dataset 
```

## Methodology

### Model Selection
We selected **Linear Regression** for its:
- Interpretability (coefficients provide clear feature importance)
- Baseline performance establishment
- Fairness analysis capabilities
- Computational efficiency

### Evaluation Metrics
- **RMSE (Root Mean Square Error):** Overall prediction accuracy
- **MAE (Mean Absolute Error):** Average prediction error
- **R² Score:** Proportion of variance explained
- **Bias:** Mean difference between predicted and actual values

## Bias Analysis Results

### Identified Biases

#### Representation Bias
- **Education Distribution:**
  - Level 1: 611 records (82.6%) - Severely over-represented
  - Level 2: 46 records (6.2%) - Under-represented
  - Level 3: 79 records (10.7%) - Under-represented
  - Level 4: 4 records (0.5%) - Extremely under-represented

- **Age Distribution:**
  - 18-30 years: 177 samples (23.9%) - Balanced
  - 31-40 years: 422 samples (57.0%) - Over-represented
  - 41-50 years: 132 samples (17.8%) - Balanced
  - 50+ years: 9 samples (1.2%) - Under-represented

#### Disproportionate Effects
- **Age-based Absenteeism:**
  - 18-30 years: 5.44 hours (below average by 1.49 hours)
  - 31-40 years: 7.06 hours (above average by 0.14 hours)
  - 41-50 years: 6.96 hours (above average by 0.04 hours)
  - 50+ years: 29.11 hours (above average by 22.19 hours)

## Corrective Measures Implemented

### 1. Feature Elimination
Removed proxy features that could encode sensitive attributes:
- **Height:** Potential proxy for gender/age
- **Weight:** Potential proxy for gender/age
- **Body Mass Index:** Derived from height/weight
- **ID:** Non-meaningful identifier

### 2. Age Group Balancing
- **Target:** Equal representation across all age groups
- **Method:** Downsampling majority groups, upsampling minority groups
- **Result:** All age groups balanced to 9 samples each

### 3. Education Level Balancing
- **Target Count:** 17 samples per education level
- **Method:** Strategic downsampling and upsampling
- **Result:** Balanced representation across education levels 1 and 3

## Results Summary

### Model Performance Comparison

| Metric | Baseline Model | Bias-Mitigated Model | Change |
|--------|----------------|---------------------|---------|
| RMSE | 11.4292 | 43.1228 | +277.4% |
| MAE | 6.4389 | 16.5046 | +156.3% |
| R² Score | -0.1987 | -0.0875 | +55.9% |

### Fairness Improvements

#### Age Group Fairness
- **Baseline:** MAE Gap = 20.78 hours
- **Mitigated:** MAE Gap = 0.00 hours (perfect equality)

#### Education Level Fairness
- **Baseline:** MAE Gap = 13.36 hours
- **Mitigated:** MAE Gap = 17.56 hours (mixed results)

#### Service Time Group Fairness
- **Baseline:** MAE Gap = 3.76 hours
- **Mitigated:** MAE Gap = 0.00 hours (perfect equality)


## Usage Instructions

### How to Run:
   ```bash
   > git clone https://github.com/krishna-kumar-bais/Fair-Absenteeism-Analysis
   
   > cd Fair-Absenteeism-Analysis

   > docker build -t absenteeism-project .

   > docker run --rm absenteeism-project
   ```

### Expected Output
The script will generate:
1. Dataset exploration and statistics
2. Bias evaluation analysis
3. Corrective measures implementation
4. Model performance comparison
5. Fairness evaluation results

## Individual Contributions

### Krishna Kumar Bais (241110038)
- Bias evaluation framework design and implementation
- Fairness metrics development for regression tasks
- Detailed statistical analysis of demographic distributions
- Age and education balancing strategies
- Model performance trade-offs analysis

### Rohan (241110057)
- Linear regression pipeline development and optimization
- Feature engineering and encoding strategies
- Comprehensive model evaluation and comparison system
- Analysis output formatting and result presentation
- Code integration and testing

## Conclusions

This analysis demonstrates the complexity of implementing fair machine learning in organizational settings. While traditional approaches like reweighting can sometimes backfire, careful feature selection and elimination of discriminatory pathways shows promise. The extreme data imbalance in our dataset highlights the critical importance of representative data collection for fair AI systems.
