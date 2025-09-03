import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')


def regression_group_fairness(y_true, y_pred, sensitive_series, group_names):
    """
    Robust version that expects:
    - y_true: pandas Series OR 1D array
    - y_pred: pandas Series OR 1D array (same length)
    - sensitive_series: pandas Series (with index corresponding to y_true/y_pred) OR array-like (aligned)
    - group_names: list/array of group labels to evaluate

    This function aligns by index when possible to avoid boolean-index length mismatch.
    """
    # Convert to pandas Series
    if not isinstance(sensitive_series, pd.Series):
        sensitive_series = pd.Series(sensitive_series)
    # If y_true/y_pred are not Series, convert them to Series
    if not isinstance(y_true, pd.Series):
        y_true = pd.Series(y_true)
    if not isinstance(y_pred, pd.Series):
        y_pred = pd.Series(y_pred)

    # Try to align by index if sensitive_series already carries index information
    # If sensitive_series has a different length than y_true, but has an index, use that index
    if len(sensitive_series) != len(y_true):
        # If sensitive_series originated from df.loc[test_idx, ...] it will be a Series and have the correct index
        if isinstance(sensitive_series, pd.Series) and not sensitive_series.index.equals(y_true.index):
            # reindex y_true and y_pred to match sensitive_series index
            try:
                y_true = y_true.reindex(sensitive_series.index)
                y_pred = y_pred.reindex(sensitive_series.index)
            except Exception:
                # fallback: truncate or pad to minimum length (less preferable)
                minlen = min(len(sensitive_series), len(y_true))
                sensitive_series = sensitive_series.iloc[:minlen]
                y_true = y_true.iloc[:minlen]
                y_pred = y_pred.iloc[:minlen]
        else:
            # fallback: align by trimming to min length
            minlen = min(len(sensitive_series), len(y_true))
            sensitive_series = sensitive_series.iloc[:minlen]
            y_true = y_true.iloc[:minlen]
            y_pred = y_pred.iloc[:minlen]

    metrics = {}
    for group in group_names:
        # mask using the sensitive_series (index-aligned)
        mask = (sensitive_series == group)
        n = mask.sum()
        if n == 0:
            continue
        true_g = y_true.loc[mask]
        pred_g = y_pred.loc[mask]
        mae = mean_absolute_error(true_g, pred_g)
        mse = mean_squared_error(true_g, pred_g)
        rmse = np.sqrt(mse)
        bias = float((pred_g - true_g).mean())  # positive => overprediction
        avg_pred = float(pred_g.mean())
        avg_true = float(true_g.mean())
        r2 = float(r2_score(true_g, pred_g)) if len(true_g) > 1 else float('nan')
        metrics[group] = {
            'count': int(n),
            'mae': float(mae),
            'rmse': float(rmse),
            'bias_mean_pred_minus_true': bias,
            'avg_pred': avg_pred,
            'avg_true': avg_true,
            'r2': r2
        }
    return metrics


def bias_evaluation(df):
    """
    Conduct comprehensive bias evaluation of the dataset
    
    Parameters:
    df: input dataframe
    
    Returns:
    dict: bias analysis results
    """
    # print("=" * 130)
    # print("BIAS EVALUATION ANALYSIS")
    # print("=" * 130)
    
    bias_analysis = {}
    
    # 1. Identify potential biases in the dataset
    print("\n1. IDENTIFYING POTENTIAL BIASES IN THE DATASET")
    print("-" * 60)
    
    # Age-based bias analysis
    print("\n1.1 AGE-BASED BIAS ANALYSIS:")
    age_groups = pd.cut(df['Age'], bins=[0, 30, 40, 50, 100], labels=['18-30', '31-40', '41-50', '50+'])
    age_distribution = age_groups.value_counts().sort_index()
    print(f"Age distribution: {age_distribution.to_dict()}")
    
    # Check if certain age groups are over/under-represented
    total_samples = len(df)
    age_bias = {}
    for age_group, count in age_distribution.items():
        percentage = (count / total_samples) * 100
        age_bias[age_group] = {
            'count': int(count),
            'percentage': float(percentage),
            'over_under_represented': 'Over-represented' if percentage > 25 else 'Under-represented' if percentage < 15 else 'Balanced'
        }
        print(f"{age_group}: {count} samples ({percentage:.1f}%) - {age_bias[age_group]['over_under_represented']}")
    
    bias_analysis['age_bias'] = age_bias
    
    # Education-based bias analysis
    print("\n1.2 EDUCATION-BASED BIAS ANALYSIS:")
    education_distribution = df['Education'].value_counts().sort_index()
    print(f"Education distribution: {education_distribution.to_dict()}")
    
    education_bias = {}
    for edu_level, count in education_distribution.items():
        percentage = (count / total_samples) * 100
        education_bias[edu_level] = {
            'count': int(count),
            'percentage': float(percentage),
            'over_under_represented': 'Over-represented' if percentage > 40 else 'Under-represented' if percentage < 10 else 'Balanced'
        }
        print(f"Education level {edu_level}: {count} samples ({percentage:.1f}%) - {education_bias[edu_level]['over_under_represented']}")
    
    bias_analysis['education_bias'] = education_bias
    
    
    # Service time bias
    print("\n1.3 SERVICE TIME BIAS ANALYSIS:")
    service_groups = pd.cut(df['Service time'], bins=[0, 5, 10, 15, 50], labels=['0-5 years', '6-10 years', '11-15 years', '15+ years'])
    service_distribution = service_groups.value_counts().sort_index()
    print(f"Service time distribution: {service_distribution.to_dict()}")
    
    # 2. Analyze disproportionate effects
    print("\n\n2. ANALYZING DISPROPORTIONATE EFFECTS")
    print("-" * 60)
    
    # Check if certain groups are disproportionately affected by absenteeism
    print("\n2.1 ABSENTEEISM IMPACT BY AGE GROUP:")
    df_with_age_groups = df.copy()
    df_with_age_groups['Age_Group'] = age_groups
    
    age_absenteeism = df_with_age_groups.groupby('Age_Group')['Absenteeism time in hours'].agg(['mean', 'std', 'count'])
    print(age_absenteeism)
    
    # Check for statistical significance (simple comparison)
    overall_mean = df['Absenteeism time in hours'].mean()
    print(f"\nOverall mean absenteeism: {overall_mean:.2f} hours")
    
    for age_group in age_absenteeism.index:
        group_mean = age_absenteeism.loc[age_group, 'mean']
        difference = group_mean - overall_mean
        print(f"{age_group}: {group_mean:.2f} hours (difference: {difference:+.2f} hours)")
    
    # Education level impact
    print("\n2.2 ABSENTEEISM IMPACT BY EDUCATION LEVEL:")
    edu_absenteeism = df.groupby('Education')['Absenteeism time in hours'].agg(['mean', 'std', 'count'])
    print(edu_absenteeism)
    
    for edu_level in edu_absenteeism.index:
        group_mean = edu_absenteeism.loc[edu_level, 'mean']
        difference = group_mean - overall_mean
        print(f"Education {edu_level}: {group_mean:.2f} hours (difference: {difference:+.2f} hours)")
    
    # 3. Identify sources of bias
    print("\n\n3. IDENTIFYING SOURCES OF BIAS")
    print("-" * 60)
    
    bias_sources = {
        'sampling_bias': 'Dataset may be collected from specific industries or regions',
        'historical_bias': 'Historical workplace patterns may favor certain demographics',
        'measurement_bias': 'Absenteeism measurement may vary across different job roles',
        'labeler_bias': 'Human labeling of absence reasons may introduce subjectivity'
    }
    
    for bias_type, description in bias_sources.items():
        print(f"{bias_type.replace('_', ' ').title()}: {description}")
    
    bias_analysis['bias_sources'] = bias_sources
    bias_analysis['age_absenteeism'] = age_absenteeism
    bias_analysis['education_absenteeism'] = edu_absenteeism
    
    return bias_analysis


def implement_corrective_measures(df, bias_analysis):
    """
    Implement corrective measures to mitigate bias
    
    Parameters:
    df: input dataframe
    bias_analysis: results from bias evaluation
    
    Returns:
    tuple: (balanced_df, corrective_measures_applied)
    """
    # print("\n" + "=" * 80)
    # print("IMPLEMENTING CORRECTIVE MEASURES")
    # print("=" * 130)
    
    corrective_measures = []
    balanced_df = df.copy()
    
    # 1. Feature elimination: Remove features that encode sensitive attributes or proxies
    print("\n1. FEATURE ELIMINATION")
    print("-" * 40)
    
    # Remove potential proxy features for sensitive attributes
    features_to_remove = []
    
    # Height and Weight can be proxies for gender/age
    if 'Height' in balanced_df.columns:
        features_to_remove.append('Height')
        print("Removed 'Height' - potential proxy for gender/age")
    
    if 'Weight' in balanced_df.columns:
        features_to_remove.append('Weight')
        print("Removed 'Weight' - potential proxy for gender/age")
    
    if 'Body mass index' in balanced_df.columns:
        features_to_remove.append('Body mass index')
        print("Removed 'Body mass index' - derived from height/weight")
    
    # Remove ID as it's not a meaningful feature
    if 'ID' in balanced_df.columns:
        features_to_remove.append('ID')
        print("Removed 'ID' - not a meaningful feature")
    
    balanced_df = balanced_df.drop(columns=features_to_remove)
    corrective_measures.append(f"Removed proxy features: {features_to_remove}")
    
    # 2. Reweighting to balance representation
    print("\n2. REWEIGHTING FOR BALANCED REPRESENTATION")
    print("-" * 40)
    
    # Create balanced dataset for age groups
    print("Balancing age group representation...")
    
    # Get the smallest age group size
    age_groups = pd.cut(balanced_df['Age'], bins=[0, 30, 40, 50, 100], labels=['18-30', '31-40', '41-50', '50+'])
    age_counts = age_groups.value_counts()
    min_age_count = age_counts.min()
    
    balanced_age_df = pd.DataFrame()
    
    for age_group in age_counts.index:
        group_data = balanced_df[age_groups == age_group]
        if len(group_data) > min_age_count:
            # Downsample larger groups
            group_data = group_data.sample(n=min_age_count, random_state=42)
        elif len(group_data) < min_age_count:
            # Upsample smaller groups
            group_data = resample(group_data, n_samples=min_age_count, random_state=42, replace=True)
        
        balanced_age_df = pd.concat([balanced_age_df, group_data])
    
    print(f"Age group balancing: All groups now have {min_age_count} samples")
    corrective_measures.append(f"Balanced age groups to {min_age_count} samples each")
    
    # 3. Resampling to balance education levels
    print("\n3. RESAMPLING FOR EDUCATION BALANCE")
    print("-" * 40)
    
    # Balance education levels (focus on main levels)
    education_counts = balanced_age_df['Education'].value_counts()
    target_education_count = education_counts.max() // 2  # Target for minority classes
    
    balanced_edu_df = pd.DataFrame()
    
    for edu_level in education_counts.index:
        edu_data = balanced_age_df[balanced_age_df['Education'] == edu_level]
        if len(edu_data) > target_education_count:
            # Downsample majority classes
            edu_data = edu_data.sample(n=target_education_count, random_state=42)
        elif len(edu_data) < target_education_count:
            # Upsample minority classes
            edu_data = resample(edu_data, n_samples=target_education_count, random_state=42, replace=True)
        
        balanced_edu_df = pd.concat([balanced_edu_df, edu_data])
    
    print(f"Education balancing: Target count per level: {target_education_count}")
    corrective_measures.append(f"Balanced education levels to target count: {target_education_count}")
    
    # 4. Final balanced dataset
    balanced_df = balanced_edu_df.copy()
    
    print(f"\nFinal balanced dataset shape: {balanced_df.shape}")
    print(f"Original dataset shape: {df.shape}")
    
    # Show the improvement in balance
    print("\nBalance improvement summary:")
    print("Age groups after balancing:")
    age_groups_balanced = pd.cut(balanced_df['Age'], bins=[0, 30, 40, 50, 100], labels=['18-30', '31-40', '41-50', '50+'])
    print(age_groups_balanced.value_counts().sort_index())
    
    print("\nEducation levels after balancing:")
    print(balanced_df['Education'].value_counts().sort_index())
    
    return balanced_df, corrective_measures


def fairness_evaluation_regression(y_true, y_pred, sensitive_attributes, model_name):
    """
    Robust fairness evaluation that aligns y_true/y_pred with sensitive attribute Series when possible.
    sensitive_attributes: dict where values can be pandas Series (preferred, with original df indices)
                          or array-like; function will attempt to reindex/align.
    """

    print(f"FAIRNESS EVALUATION - {model_name.upper()}")
    print("-" * 40)
    
    # Convert overall arrays to Series (no index yet)
    if not isinstance(y_true, pd.Series):
        y_true = pd.Series(y_true)
    if not isinstance(y_pred, pd.Series):
        y_pred = pd.Series(y_pred)
    
    fairness_results = {}
    
    # Overall metrics (align lengths if needed)
    minlen = min(len(y_true), len(y_pred))
    if len(y_true) != len(y_pred):
        y_true = y_true.iloc[:minlen]
        y_pred = y_pred.iloc[:minlen]
    
    overall_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    overall_mae = mean_absolute_error(y_true, y_pred)
    overall_r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float('nan')
    
    print(f"OVERALL MODEL PERFORMANCE (REGRESSION):")
    print(f"RMSE: {overall_rmse:.4f}")
    print(f"MAE: {overall_mae:.4f}")
    print(f"R² Score: {overall_r2:.4f}")
    
    fairness_results['overall'] = {
        'rmse': float(overall_rmse),
        'mae': float(overall_mae),
        'r2': float(overall_r2)
    }
    
    for attr_name, attr_values in sensitive_attributes.items():
        print("\n")
        print(f"FAIRNESS EVALUATION ACROSS {attr_name.upper()}")
        print(f"{'-' * 40}")
        
        # Keep the original Series if provided, otherwise convert (will have default RangeIndex)
        if isinstance(attr_values, pd.Series):
            attr_series = attr_values
        else:
            attr_series = pd.Series(attr_values)
        
        # If attr_series has an index that doesn't match y_true's, reindex y_true and y_pred to attr_series.index
        if not attr_series.index.equals(y_true.index):
            # try to reindex y_true/pred to attr_series index
            try:
                y_true_aligned = y_true.reindex(attr_series.index)
                y_pred_aligned = y_pred.reindex(attr_series.index)
                # drop NaNs that came from reindexing
                valid_mask = y_true_aligned.notna() & y_pred_aligned.notna() & attr_series.notna()
                y_true_aligned = y_true_aligned.loc[valid_mask]
                y_pred_aligned = y_pred_aligned.loc[valid_mask]
                attr_series = attr_series.loc[valid_mask]
            except Exception:
                # fallback: truncate to min length
                minlen = min(len(attr_series), len(y_true))
                attr_series = attr_series.iloc[:minlen]
                y_true_aligned = y_true.iloc[:minlen]
                y_pred_aligned = y_pred.iloc[:minlen]
        else:
            y_true_aligned = y_true
            y_pred_aligned = y_pred
            # also remove NaNs in attr if any
            valid_mask = attr_series.notna()
            y_true_aligned = y_true_aligned.loc[valid_mask]
            y_pred_aligned = y_pred_aligned.loc[valid_mask]
            attr_series = attr_series.loc[valid_mask]
        
        unique_values = np.unique(attr_series)
        group_metrics = regression_group_fairness(y_true_aligned, y_pred_aligned, attr_series, unique_values)
        
        print(f"Performance by {attr_name}:")
        print(f"{'Group':<20} {'Count':<6} {'MAE':<10} {'RMSE':<10} {'Bias':<12} {'AvgPred':<10} {'AvgTrue':<10} {'R2':<8}")
        print("-" * 95)
        for group, metrics in group_metrics.items():
            print(f"{str(group):<20} {metrics['count']:<6} {metrics['mae']:<10.4f} {metrics['rmse']:<10.4f} {metrics['bias_mean_pred_minus_true']:<12.4f} "
                  f"{metrics['avg_pred']:<10.3f} {metrics['avg_true']:<10.3f} {metrics['r2']:<8.4f}")
        
        # gaps
        maes = [m['mae'] for m in group_metrics.values()]
        rmses = [m['rmse'] for m in group_metrics.values()]
        biases = [m['bias_mean_pred_minus_true'] for m in group_metrics.values()]
        avg_preds = [m['avg_pred'] for m in group_metrics.values()]
        
        if len(maes) > 1:
            mae_gap = max(maes) - min(maes)
            rmse_gap = max(rmses) - min(rmses)
            bias_gap = max(biases) - min(biases)
            pred_gap = max(avg_preds) - min(avg_preds)
        else:
            mae_gap = rmse_gap = bias_gap = pred_gap = 0.0
        
        print(f"\nGroup gaps for {attr_name}: MAE_gap={mae_gap:.4f}, RMSE_gap={rmse_gap:.4f}, Bias_gap={bias_gap:.4f}, AvgPred_gap={pred_gap:.4f}")
        fairness_results[attr_name] = {
            'group_metrics': group_metrics,
            'mae_gap': float(mae_gap),
            'rmse_gap': float(rmse_gap),
            'bias_gap': float(bias_gap),
            'avg_pred_gap': float(pred_gap)
        }
    return fairness_results


def main():
    # 1. Data Loading and Initial Exploration
    print("\n")
    print("=" * 130)
    print("1. DATA LOADING AND INITIAL EXPLORATION")
    print("=" * 130)
    # print("\n")
    
    # Load the dataset
    df = pd.read_csv('Absenteeism_at_work.csv', sep=';')
    print(f"Dataset shape: {df.shape}")
    print(f"First few rows:")
    print(df.head())
    
    # Basic information about the dataset
    print(f"\nDataset Info:")
    print(f"Number of records: {df.shape[0]}")
    print(f"Number of features: {df.shape[1]}")
    print(f"\nColumns:")
    for i, col in enumerate(df.columns):
        print(f"{i+1:2d}. {col}")
    print(f"\nData types:")
    print(df.dtypes)
    
    # 2. Data Understanding and Preprocessing
    print("\n")
    print("=" * 130)
    print("2. DATA UNDERSTANDING AND PREPROCESSING")
    print("=" * 130)
    # print("\n")
    
    # Check for missing values
    print("Missing values in each column:")
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(missing_values[missing_values > 0])
    else:
        print("No missing values found")
    
    # Check for duplicate records
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate records: {duplicates}")
    
    # Basic statistics
    print("\nBasic statistics:")
    print(df.describe())
    
    # Understanding the target variable
    print(f"\nTarget variable analysis:")
    print(f"Target variable: Absenteeism time in hours")
    print(f"Range: {df['Absenteeism time in hours'].min()} to {df['Absenteeism time in hours'].max()} hours")
    print(f"Mean: {df['Absenteeism time in hours'].mean():.2f} hours")
    print(f"Median: {df['Absenteeism time in hours'].median():.2f} hours")
    print(f"Standard deviation: {df['Absenteeism time in hours'].std():.2f} hours")
    
    # 3. Feature Analysis
    print("\n")
    print("=" * 130)
    print("3. FEATURE ANALYSIS")
    print("=" * 130)
    # print("\n")
    
    # Analyze categorical variables
    categorical_cols = ['Reason for absence', 'Month of absence', 'Day of the week', 'Seasons', 
                        'Hit target', 'Disciplinary failure', 'Education', 'Son', 
                        'Social drinker', 'Social smoker', 'Pet']
    
    print("Categorical variables analysis:")
    for col in categorical_cols:
        if col in df.columns:
            print(f"\n{col}:")
            print(f"Unique values: {df[col].nunique()}")
            print(f"Value counts:")
            print(df[col].value_counts().head(5))
    
    # Analyze numerical variables
    numerical_cols = ['Transportation expense', 'Distance from Residence to Work', 'Service time', 
                      'Age', 'Work load Average/day ', 'Weight', 'Height', 'Body mass index']
    
    print(f"\nNumerical variables analysis:")
    
    # Create correlation matrix for numerical variables
    numerical_df = df[numerical_cols + ['Absenteeism time in hours']]
    correlation_matrix = numerical_df.corr()
    
    print("\nCorrelation with target variable (Absenteeism time in hours):")
    target_corr = correlation_matrix['Absenteeism time in hours'].sort_values(ascending=False)
    print(target_corr)
    
    # 4. BIAS EVALUATION (NEW SECTION)
    print("\n")
    print("=" * 130)
    print("4. BIAS EVALUATION")
    print("=" * 130)
    # print("\n")
    
    # Conduct comprehensive bias evaluation
    bias_analysis = bias_evaluation(df)
    
    # ---------- BASELINE MODEL (LINEAR REGRESSION ON ORIGINAL DATA) ----------
    print("\n")
    print("=" * 130)
    print("5. BASELINE MODEL (LINEAR REGRESSION ON ORIGINAL DATA)")
    print("=" * 130)
    # print("\n")

    # Work on a copy
    df_base = df.copy()

    # Prepare categorical columns for one-hot encoding (only if exist)
    available_cat = [c for c in categorical_cols if c in df_base.columns]
    df_base_encoded = pd.get_dummies(df_base, columns=available_cat, drop_first=True)

    # Regression baseline: predict continuous absenteeism
    X_base_reg = df_base_encoded.drop(['Absenteeism time in hours'], axis=1)
    y_base_reg = df_base_encoded['Absenteeism time in hours']

    # Train/test split
    Xb_train_r, Xb_test_r, yb_train_r, yb_test_r = train_test_split(X_base_reg, y_base_reg, test_size=0.2, random_state=42)

    # Scale features
    scaler_base = StandardScaler()
    Xb_train_r_scaled = scaler_base.fit_transform(Xb_train_r)
    Xb_test_r_scaled = scaler_base.transform(Xb_test_r)

    # Linear Regression baseline
    lr_base = LinearRegression()
    lr_base.fit(Xb_train_r_scaled, yb_train_r)
    yb_pred_lr_base = lr_base.predict(Xb_test_r_scaled)
    mse_lr_base = mean_squared_error(yb_test_r, yb_pred_lr_base)
    rmse_lr_base = np.sqrt(mse_lr_base)
    mae_lr_base = mean_absolute_error(yb_test_r, yb_pred_lr_base)
    r2_lr_base = r2_score(yb_test_r, yb_pred_lr_base)

    # print("\nLinear Regression (baseline) — Regression metrics")
    # print(f"MSE: {mse_lr_base:.4f}, RMSE: {rmse_lr_base:.4f}, MAE: {mae_lr_base:.4f}, R2: {r2_lr_base:.4f}")

    # Fairness evaluation (regression) for baseline model
    test_idx_base = Xb_test_r.index
    sensitive_attrs_base = {}
    if 'Age' in df_base.columns:
        age_groups_base = pd.cut(df_base.loc[test_idx_base, 'Age'], bins=[0, 30, 40, 50, 100], labels=['18-30', '31-40', '41-50', '50+'])
        sensitive_attrs_base['age_group'] = age_groups_base.values
    if 'Education' in df_base.columns:
        sensitive_attrs_base['education_level'] = df_base.loc[test_idx_base, 'Education'].values
    if 'Service time' in df_base.columns:
        service_groups_base = pd.cut(df_base.loc[test_idx_base, 'Service time'], bins=[0, 5, 10, 15, 50], labels=['0-5 years', '6-10 years', '11-15 years', '15+ years'])
        sensitive_attrs_base['service_time_group'] = service_groups_base.values

    # print("\nBaseline Fairness Evaluation (Regression): Linear Regression (baseline)")
    fairness_base_lr = fairness_evaluation_regression(yb_test_r.values, yb_pred_lr_base, sensitive_attrs_base, "Linear Regression (Baseline)")

    # ---------- END BASELINE SECTION ----------

    # 6. IMPLEMENTING CORRECTIVE MEASURES (NEW SECTION)
    print("\n")
    print("=" * 130)
    print("6. IMPLEMENTING CORRECTIVE MEASURES")
    print("=" * 130)
    # print("\n")
    
    # Implement corrective measures to mitigate bias
    balanced_df, corrective_measures = implement_corrective_measures(df, bias_analysis)
    
    print(f"\nCorrective measures applied:")
    for i, measure in enumerate(corrective_measures, 1):
        print(f"{i}. {measure}")
    
    # 7. Data Preprocessing for Machine Learning (Updated with balanced data)
    print("\n")
    print("=" * 130)
    print("7. DATA PREPROCESSING FOR MACHINE LEARNING (WITH BIAS MITIGATION)")
    print("=" * 130)
    # print("\n")
    
    # Use balanced dataset for preprocessing
    df_processed = balanced_df.copy()
    
    # Handle categorical variables
    categorical_cols = ['Reason for absence', 'Month of absence', 'Day of the week', 'Seasons', 
                        'Hit target', 'Disciplinary failure', 'Education', 'Son', 
                        'Social drinker', 'Social smoker', 'Pet']
    
    # Create dummy variables for categorical columns (only those that exist)
    available_cat = [c for c in categorical_cols if c in df_processed.columns]
    df_encoded = pd.get_dummies(df_processed, columns=available_cat, drop_first=True)
    
    print(f"Balanced dataset shape: {balanced_df.shape}")
    print(f"After encoding shape: {df_encoded.shape}")
    print(f"New columns created: {df_encoded.shape[1] - balanced_df.shape[1]}")
    
    # Prepare features and target
    X = df_encoded.drop(['Absenteeism time in hours'], axis=1)
    y = df_encoded['Absenteeism time in hours']
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Features scaled successfully")
    
    # 8. REGRESSION MODELS (WITH BIAS MITIGATION) - ONLY LINEAR REGRESSION
    print("\n")
    print("=" * 130)
    print("8. REGRESSION MODEL FOR PREDICTING ABSENTEEISM TIME (WITH BIAS MITIGATION)")
    print("=" * 130)
    # print("\n")
    
    # Linear Regression (bias-mitigated)
    print("Linear Regression Model (Bias-Mitigated):")
    print("-" * 50)
    
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_lr = lr_model.predict(X_test_scaled)
    
    # Evaluation metrics
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mse_lr)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)
    
    print(f"Mean Squared Error: {mse_lr:.4f}")
    print(f"Root Mean Squared Error: {rmse_lr:.4f}")
    print(f"Mean Absolute Error: {mae_lr:.4f}")
    print(f"R² Score: {r2_lr:.4f}")
    
    # Feature importance approximation for linear regression via coefficients
    feature_importance_lr = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': lr_model.coef_
    })
    feature_importance_lr = feature_importance_lr.reindex(feature_importance_lr['Coefficient'].abs().sort_values(ascending=False).index)
    
    
    # 9. FAIRNESS EVALUATION (REGRESSION) FOR BIAS-MITIGATED MODEL
    print("\n")
    print("=" * 130)
    print("9. FAIRNESS EVALUATION (REGRESSION)")
    print("=" * 130)
    # print("\n")
    
    # Prepare sensitive attributes for fairness evaluation using test indices
    test_indices = X_test.index
    sensitive_attributes = {}
    
    # Age groups
    if 'Age' in df_processed.columns:
        age_groups = pd.cut(df_processed.loc[test_indices, 'Age'], bins=[0, 30, 40, 50, 100], labels=['18-30', '31-40', '41-50', '50+'])
        sensitive_attributes['age_group'] = age_groups.values
    
    # Education levels
    if 'Education' in df_processed.columns:
        sensitive_attributes['education_level'] = df_processed.loc[test_indices, 'Education'].values
    
    # Service time groups
    if 'Service time' in df_processed.columns:
        service_groups = pd.cut(df_processed.loc[test_indices, 'Service time'], bins=[0, 5, 10, 15, 50], labels=['0-5 years', '6-10 years', '11-15 years', '15+ years'])
        sensitive_attributes['service_time_group'] = service_groups.values
    
    fairness_lr = fairness_evaluation_regression(y_test.values, y_pred_lr, sensitive_attributes, "Linear Regression (Bias-Mitigated)")
    
    # 10. FINAL SUMMARY AND COMPARISON
    print("\n")
    print("=" * 130)
    print("10. FINAL ANALYSIS SUMMARY AND FAIRNESS COMPARISON")
    print("=" * 130)
    # print("\n")

    print(f"Dataset: Absenteeism at Work")
    print(f"Original samples: {df.shape[0]}")
    print(f"Balanced samples: {balanced_df.shape[0]}")
    print(f"Target variable: Absenteeism time in hours")
    print(f"Target range: {df['Absenteeism time in hours'].min()} - {df['Absenteeism time in hours'].max()} hours")
    print(f"Target mean: {df['Absenteeism time in hours'].mean():.2f} hours")
    
    print(f"\nREGRESSION MODELS:")
    print(f"Baseline Linear Regression - RMSE: {rmse_lr_base:.4f}, MAE: {mae_lr_base:.4f}, R2: {r2_lr_base:.4f}")
    print(f"Bias-mitigated Linear Regression - RMSE: {rmse_lr:.4f}, MAE: {mae_lr:.4f}, R2: {r2_lr:.4f}")
    
    print(f"\nFAIRNESS COMPARISON (example: education_level MAE gap):")
    edu_gap_base = fairness_base_lr.get('education_level', {}).get('mae_gap', None) if isinstance(fairness_base_lr, dict) else None
    edu_gap_mitigated = fairness_lr.get('education_level', {}).get('mae_gap', None) if isinstance(fairness_lr, dict) else None
    print(f"Baseline MAE gap (education): {edu_gap_base}")
    print(f"After mitigation MAE gap (education): {edu_gap_mitigated}")
    
    print(f"\nKEY INSIGHTS:")
    print(f"1. Model used: Linear Regression (baseline and bias-mitigated)")
    print(f"2. Bias mitigation applied: {len(corrective_measures)} measures")
    print(f"3. Fairness evaluation completed across selected sensitive attributes")
    print(f"\nAnalysis completed successfully with Linear Regression focused evaluation!")

if __name__ == "__main__":
    main()