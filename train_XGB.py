import pandas as pd
import arff
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import optuna
import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, make_scorer


data_freq = arff.load('freMTPL2freq.arff')
df_freq = pd.DataFrame(data_freq, columns=["IDpol", "ClaimNb", "Exposure", "Area", "VehPower", "VehAge","DrivAge", "BonusMalus", "VehBrand", "VehGas", "Density", "Region"])
data_sev = arff.load('freMTPL2sev.arff')
df_sev = pd.DataFrame(data_sev, columns=["IDpol", "ClaimAmount"])
del data_freq, data_sev

input_standardization = False
output_standardization = False

def objective(trial):
    # Load sample data
    # First sum up the "ClaimAmount" with duplicate entries of "IDpol".
    df_sev.groupby("IDpol").sum()
    # There are some nan features in df_freq with some "IDpol" in df_freq, which is directly cleaned,
    merged_df = pd.merge(df_freq, df_sev, on="IDpol", how="inner")

    input_scalers_pred_claimamount = []
    output_scalers_pred_claimamount = []

    for column in merged_df.columns:
      if column == "IDpol": # Remove the feature 'IDpol'
        merged_df.drop(column, axis=1)
      else:
        if merged_df[column].dtype == 'object':
          # Apply one-hot encoding to 'object' type columns
          dummies = pd.get_dummies(merged_df[column], prefix=column)
          merged_df = pd.concat([merged_df.drop(column, axis=1), dummies], axis=1)
        elif merged_df[column].dtype == np.float64:
          if input_standardization and column != 'Exposure':
            # Normalize 'float' type columns
            scaler = preprocessing.StandardScaler()
            merged_df[column] = scaler.fit_transform(merged_df[[column]])
            input_scalers_pred_claimamount.append(scaler)
          elif output_standardization and column == 'Exposure':
            scaler = preprocessing.StandardScaler()
            merged_df[column] = scaler.fit_transform(merged_df[[column]])
            output_scalers_pred_claimamount.append(scaler)

    y = np.asarray(merged_df["ClaimAmount"])
    X = np.asarray(merged_df.drop("ClaimAmount", axis=1))

    # Define the hyperparameter configuration space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 400),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-5, 1.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-5, 1.0)
    }

    # Initialize XGBoost regressor
    regressor = xgb.XGBRegressor(**params, objective='reg:squarederror', random_state=42)

    # Perform 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(regressor, X, y, cv=kf, scoring=make_scorer(mean_squared_error))
    print(scores)
    # Compute RMSE from scores
    rmse = (scores.mean())**0.5
    return rmse

# Create a study object and specify the direction of the optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=2)

# Print the best parameters
print('Best trial:', study.best_trial)
print('Best parameters:', study.best_params)
# feature preprocessing

# Best trial: FrozenTrial(number=10, state=1, values=[29492.26031002925], datetime_start=datetime.datetime(2024, 4, 27, 15, 32, 23, 946406), datetime_complete=datetime.datetime(2024, 4, 27, 15, 32, 24, 997668), params={'n_estimators': 57, 'max_depth': 10, 'learning_rate': 0.010226832085816177, 'subsample': 0.8436694535406467, 'colsample_bytree': 0.7803365182612978, 'min_child_weight': 6, 'reg_alpha': 1.1464156634946757e-05, 'reg_lambda': 1.2490737133035235e-05}, user_attrs={}, system_attrs={}, intermediate_values={}, distributions={'n_estimators': IntDistribution(high=400, log=False, low=50, step=1), 'max_depth': IntDistribution(high=20, log=False, low=3, step=1), 'learning_rate': FloatDistribution(high=0.3, log=True, low=0.01, step=None), 'subsample': FloatDistribution(high=1.0, log=False, low=0.5, step=None), 'colsample_bytree': FloatDistribution(high=1.0, log=False, low=0.5, step=None), 'min_child_weight': IntDistribution(high=10, log=False, low=1, step=1), 'reg_alpha': FloatDistribution(high=1.0, log=True, low=1e-05, step=None), 'reg_lambda': FloatDistribution(high=1.0, log=True, low=1e-05, step=None)}, trial_id=10, value=None)
# Best parameters: {'n_estimators': 57, 'max_depth': 10, 'learning_rate': 0.010226832085816177, 'subsample': 0.8436694535406467, 'colsample_bytree': 0.7803365182612978, 'min_child_weight': 6, 'reg_alpha': 1.1464156634946757e-05, 'reg_lambda': 1.2490737133035235e-05}

# Visualize from XGBoostRegressor
import matplotlib.pyplot as plt

# Plotting the feature importance
xgb.plot_importance(model, importance_type='weight', title='Feature Importance based on weight')
plt.show()

xgb.plot_importance(model, importance_type='gain', title='Feature Importance based on gain')
plt.show()

xgb.plot_importance(model, importance_type='cover', title='Feature Importance based on cover')
plt.show()