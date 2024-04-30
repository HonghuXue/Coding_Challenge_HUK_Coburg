# To add for hyperparmaeter tunning in XGBoost: 'tweedie_variance_power':1.6

import pandas as pd
import arff
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import optuna
import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, make_scorer, mean_absolute_error
import matplotlib.pyplot as plt
import torch

data_freq = arff.load('freMTPL2freq.arff')
df_freq = pd.DataFrame(data_freq, columns=["IDpol", "ClaimNb", "Exposure", "Area", "VehPower", "VehAge", "DrivAge",
                                           "BonusMalus", "VehBrand", "VehGas", "Density", "Region"])
data_sev = arff.load('freMTPL2sev.arff')
df_sev = pd.DataFrame(data_sev, columns=["IDpol", "ClaimAmount"])
del data_freq, data_sev
merge_by_intersection = False


# First sum up the "ClaimAmount" with duplicate entries of "IDpol".
df_sev = df_sev.groupby("IDpol", as_index=False).agg({'ClaimAmount': 'sum'})

# There are some nan features in df_freq with some "IDpol" in df_freq, which is directly cleaned,
if merge_by_intersection:
    merged_df = pd.merge(df_freq, df_sev, on="IDpol", how="inner")
else:
    merged_df = pd.merge(df_freq, df_sev, on="IDpol", how="outer")
    merged_df["ClaimAmount"] = merged_df["ClaimAmount"].fillna(0)
    merged_df = merged_df.dropna(how="any")
merged_df = merged_df.drop(['IDpol', 'ClaimNb'], axis=1)
input_scalers_pred_claimamount = []
output_scalers_pred_claimamount = []

input_standardization = True
output_standardization = False
manual_one_hot_encoding = False

merged_df['ClaimAmount'] = merged_df['ClaimAmount'] / merged_df["Exposure"]


for column in merged_df.columns:
    if merged_df[column].dtype == 'object' and manual_one_hot_encoding:
        # Apply one-hot encoding to 'object' type columns
        dummies = pd.get_dummies(merged_df[column], prefix=column)
        merged_df = pd.concat([merged_df.drop(column, axis=1), dummies], axis=1)
    elif merged_df[column].dtype == np.float64:
        if input_standardization and column != 'Exposure' and column != 'ClaimAmount':
            # Normalize 'float' type columns
            scaler = preprocessing.MinMaxScaler()
            merged_df[column] = scaler.fit_transform(merged_df[[column]])
            input_scalers_pred_claimamount.append(scaler)
        elif output_standardization and column == 'ClaimAmount':
            scaler = preprocessing.MinMaxScaler()
            merged_df[column] = scaler.fit_transform(merged_df[[column]])
            output_scalers_pred_claimamount.append(scaler)

y = np.asarray(merged_df.pop("ClaimAmount"))

if manual_one_hot_encoding:
    X = np.asarray(merged_df.drop(['Exposure'], axis=1))
else:
    X = merged_df.drop(['Exposure'], axis=1)
    X["Area"] = X["Area"].astype("category")
    X["VehGas"] = X["VehGas"].astype("category")
    X["VehBrand"] = X["VehBrand"].astype("category")
    X["Region"] = X["Region"].astype("category")


def objective(trial, X, y):
    # Load sample data
    # Define the hyperparameter configuration space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 150),
        'max_depth': trial.suggest_int('max_depth', 15 if manual_one_hot_encoding else 3, 50 if manual_one_hot_encoding else 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1.0, log=True)
    }

    # Initialize XGBoost regressor
    if manual_one_hot_encoding:
        regressor = xgb.XGBRegressor(**params, objective='reg:tweedie', random_state=42, device="cuda" if torch.cuda.is_available() else "cpu")
    else:
        regressor = xgb.XGBRegressor(**params, objective='reg:tweedie', random_state=42, tree_method="hist", enable_categorical=True,
                                     device="cuda" if torch.cuda.is_available() else "cpu")

    # Perform 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(regressor, X, y, cv=kf, scoring=make_scorer(mean_squared_error))
    # Compute RMSE from scores
    rmse = (scores.mean())**0.5
    return rmse

# Create a study object and specify the direction of the optimization
study = optuna.create_study(direction='minimize')

def objective_wrapper(trial):
    return objective(trial, X, y)

study.optimize(objective_wrapper, n_trials=30)

# Print the best parameters
print('Best trial:', study.best_trial)
print('Best parameters:', study.best_params)
# feature preprocessing

# Best trial: FrozenTrial(number=10, state=1, values=[29492.26031002925], datetime_start=datetime.datetime(2024, 4, 27, 15, 32, 23, 946406), datetime_complete=datetime.datetime(2024, 4, 27, 15, 32, 24, 997668), params={'n_estimators': 57, 'max_depth': 10, 'learning_rate': 0.010226832085816177, 'subsample': 0.8436694535406467, 'colsample_bytree': 0.7803365182612978, 'min_child_weight': 6, 'reg_alpha': 1.1464156634946757e-05, 'reg_lambda': 1.2490737133035235e-05}, user_attrs={}, system_attrs={}, intermediate_values={}, distributions={'n_estimators': IntDistribution(high=400, log=False, low=50, step=1), 'max_depth': IntDistribution(high=20, log=False, low=3, step=1), 'learning_rate': FloatDistribution(high=0.3, log=True, low=0.01, step=None), 'subsample': FloatDistribution(high=1.0, log=False, low=0.5, step=None), 'colsample_bytree': FloatDistribution(high=1.0, log=False, low=0.5, step=None), 'min_child_weight': IntDistribution(high=10, log=False, low=1, step=1), 'reg_alpha': FloatDistribution(high=1.0, log=True, low=1e-05, step=None), 'reg_lambda': FloatDistribution(high=1.0, log=True, low=1e-05, step=None)}, trial_id=10, value=None)
# Best parameters: {'n_estimators': 57, 'max_depth': 10, 'learning_rate': 0.010226832085816177, 'subsample': 0.8436694535406467, 'colsample_bytree': 0.7803365182612978, 'min_child_weight': 6, 'reg_alpha': 1.1464156634946757e-05, 'reg_lambda': 1.2490737133035235e-05}

# ----------------------------Train the XGBoostRegressor with Optuna Hyper-parameters-----------------------------
if manual_one_hot_encoding: #reg:tweedie   reg:absoluteerror
    model = xgb.XGBRegressor(**study.best_params, objective='reg:tweedie', random_state=42, device="cuda" if torch.cuda.is_available() else "cpu")
else:
    model = xgb.XGBRegressor(**study.best_params, objective='reg:tweedie', random_state=42, tree_method="hist", enable_categorical=True,
                                     device="cuda" if torch.cuda.is_available() else "cpu")
model.fit(X, y)


# ----------------------------Visualize the feature importance from XGBoostRegressor-----------------------------
if manual_one_hot_encoding:
    merged_df = merged_df.drop('Exposure', axis = 1)
    feature_names = list(merged_df.columns.values) # ['VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Density', "Area_'A'", "Area_'B'", "Area_'C'", "Area_'D'", "Area_'E'", "Area_'F'", "VehBrand_'B1'", "VehBrand_'B10'", "VehBrand_'B11'", "VehBrand_'B12'", "VehBrand_'B13'", "VehBrand_'B14'", "VehBrand_'B2'", "VehBrand_'B3'", "VehBrand_'B4'", "VehBrand_'B5'", "VehBrand_'B6'", 'VehGas_Diesel', 'VehGas_Regular', "Region_'R11'", "Region_'R21'", "Region_'R22'", "Region_'R23'", "Region_'R24'", "Region_'R25'", "Region_'R26'", "Region_'R31'", "Region_'R41'", "Region_'R42'", "Region_'R43'", "Region_'R52'", "Region_'R53'", "Region_'R54'", "Region_'R72'", "Region_'R73'", "Region_'R74'", "Region_'R82'", "Region_'R83'", "Region_'R91'", "Region_'R93'", "Region_'R94'"]
    # Update feature names
    model.get_booster().feature_names = feature_names

    importances = model.get_booster().get_score(importance_type='weight')
    gains = model.get_booster().get_score(importance_type='gain')
    covers = model.get_booster().get_score(importance_type='cover')
    # Group names by their prefixes (assuming they follow a consistent naming convention)
    groups = {
        'Area': [],
        'VehBrand': [],
        'VehGas': [],
        'Region': []
    }

    # Fill groups with feature names based on the prefix
    for feature in model.get_booster().feature_names:
        for key in groups.keys():
            if feature.startswith(key):
                groups[key].append(feature)

    # Calculate the mean importance for each group and create a new importance dictionary

    # Feature importance measure from XGBoost :  https://www.restack.io/docs/mlflow-knowledge-feature-importance-xgboost-mlflow
    # Weight: The number of times a feature appears in a tree across the ensemble of trees.
    # Gain: The average gain of splits that use the feature.
    # Cover: The average coverage of splits that use the feature.


    mean_importances = {}
    mean_gains={}
    mean_covers={}
    for group, features in groups.items():
        if features:  # Ensure there are features in the group
            group_importance = sum(importances.get(f, 0) for f in features) / len(features)
            mean_importances[group] = group_importance
            group_gain = sum(gains.get(f, 0) for f in features) / len(features)
            mean_gains[group] = group_gain
            group_cover = sum(covers.get(f, 0) for f in features) / len(features)
            mean_covers[group] = group_cover

    # Also keep the non-grouped features
    for feature in importances:
        if all(not feature.startswith(key) for key in groups):
            mean_importances[feature] = importances[feature]
            mean_gains[feature] = gains[feature]
            mean_covers[feature] = covers[feature]

    # sort
    sorted_features = sorted(mean_importances, key=mean_importances.get, reverse=True)
    sorted_importances = [mean_importances[feature] for feature in sorted_features]


    plt.figure(figsize=(12, 8))
    plt.barh(sorted_features, sorted_importances, color='skyblue')
    plt.xlabel('Mean Importance')
    plt.title('Aggregated Feature Importance')
    plt.gca().invert_yaxis()  # Invert y axis for top-to-bottom descending order
    plt.show()

    sorted_features = sorted(mean_gains, key=mean_gains.get, reverse=True)
    sorted_gains = [mean_gains[feature] for feature in sorted_features]

    plt.figure(figsize=(12, 8))
    plt.barh(sorted_features, sorted_gains, color='skyblue')
    plt.xlabel('Mean gains')
    plt.title('Aggregated Feature Gains')
    plt.gca().invert_yaxis()  # Invert y axis for top-to-bottom descending order
    plt.show()

    sorted_features = sorted(mean_covers, key=mean_covers.get, reverse=True)
    sorted_covers = [mean_covers[feature] for feature in sorted_features]

    plt.figure(figsize=(12, 8))
    plt.barh(sorted_features, sorted_covers, color='skyblue')
    plt.xlabel('Mean covers')
    plt.title('Aggregated Feature Covers')
    plt.gca().invert_yaxis()  # Invert y axis for top-to-bottom descending order
    plt.show()


else:
    # # Get a graph
    graph = xgb.to_graphviz(model, num_trees=1)
    # # Or get a matplotlib axis
    ax = xgb.plot_tree(model, num_trees=1)
    # Get feature importances
    # print(model.feature_importances_)

    # visualize the tree
    # xgb.plot_tree(model, num_trees=0)


    # # Plotting the feature importance
    xgb.plot_importance(model, importance_type='weight', title='Feature Importance based on weight')
    plt.show()

    xgb.plot_importance(model, importance_type='gain', title='Feature Importance based on gain')
    plt.show()

    xgb.plot_importance(model, importance_type='cover', title='Feature Importance based on cover')
    plt.show()