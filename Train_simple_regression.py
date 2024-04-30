import pandas as pd
import arff
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, TweedieRegressor
from sklearn import linear_model
import optuna
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, make_scorer, mean_absolute_error
import seaborn as sns


random_seed = 35
np.random.seed(random_seed)
feature_visualization = True


data_freq = arff.load('freMTPL2freq.arff')
df_freq = pd.DataFrame(data_freq, columns=["IDpol", "ClaimNb", "Exposure", "Area", "VehPower", "VehAge","DrivAge", "BonusMalus", "VehBrand", "VehGas", "Density", "Region"])
data_sev = arff.load('freMTPL2sev.arff')
df_sev = pd.DataFrame(data_sev, columns=["IDpol", "ClaimAmount"])
del data_freq, data_sev
# print(np.unique(df_sev["ClaimAmount"])) # The minimum of Claimamount is 1, whereas the maximum is 4e6.
print(np.unique(df_freq["BonusMalus"], return_counts= True))


# Hyper-parameters
input_standardization = True
output_standardization = False
merge_by_intersection = False

# First sum up the "ClaimAmount" with duplicate entries of "IDpol".
# df_sev = df_sev.groupby("IDpol").sum()
df_sev = df_sev.groupby("IDpol", as_index=False).agg({'ClaimAmount': 'sum'})

# There are some nan features in df_freq with some "IDpol" in df_freq, which is directly cleaned,
if merge_by_intersection:
    merged_df = pd.merge(df_freq, df_sev, on="IDpol", how="inner")
else:
    merged_df = pd.merge(df_freq, df_sev, on="IDpol", how="outer")
    merged_df["ClaimAmount"] = merged_df["ClaimAmount"].fillna(0)
    merged_df = merged_df.dropna(how="any")


merged_df = merged_df.drop(['IDpol','ClaimNb'], axis = 1)

# input_scalers_pred_claimamount = []
# output_scalers_pred_claimamount = []

#----------------visualize Features----------------------
categorical_columns = ['Area', 'VehBrand', 'VehGas', 'Region']
numerical_columns = [col for col in df_freq.columns if col not in categorical_columns]

def plot_data_distribution(df):
    for column in df.columns:
        plt.figure(figsize=(10, 4))  # Set figure size

        # Check if the column is categorical
        if column in categorical_columns:
            # Bar plot for categorical data
            sns.countplot(data=df, x=column)
            plt.title(f'Bar Plot of {column}')
            plt.xticks(rotation=45)  # Rotate x labels for better visibility if needed
        else:
            # Histogram for numerical data
            sns.histplot(data=df, x=column, kde=False, bins=30)  # KDE for smooth distribution curve
            plt.title(f'Histogram of {column}')

        plt.ylabel('Count')
        plt.xlabel(column)
        plt.grid(True)
        plt.show()


if feature_visualization:
    merged_df['ClaimAmount'] = merged_df['ClaimAmount'] / merged_df['Exposure']
    column = 'ClaimAmount'
    plt.figure(figsize=(10, 4))  # Set figure size
    sns.histplot(data=merged_df, x=column, kde=False, bins=50)  # KDE for smooth distribution curve
    plt.title(f'Histogram of {column}')
    plt.ylabel('Count')
    plt.xlabel(column)
    plt.yscale('log')
    plt.grid(True)
    plt.show()

    # plot_data_distribution(merged_df)
#----------------visualize Features----------------------




# Define the column transformer for handling different types of data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', preprocessing.MinMaxScaler(), [col for col in merged_df.columns if merged_df[col].dtype in [np.float64, np.float32] and col not in ['ClaimAmount', 'Exposure', 'IDpol']]),
        ('cat', preprocessing.OneHotEncoder(handle_unknown='ignore'), [col for col in merged_df.columns if merged_df[col].dtype == 'object']),
    ],
    remainder='passthrough'
)

# Exclude 'ClaimAmount' and 'Exposure' from any transformation if output_standardization is applied
if output_standardization:
    output_scaler = preprocessing.MinMaxScaler()
claim_amount = merged_df.pop('ClaimAmount') #/ merged_df.pop('Exposure')  # Separate and transform this column separately
print(np.min(claim_amount))
# Create a processing pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    # ('output_scaler', output_scaler) if output_standardization else ('dummy', 'passthrough')
])

# Fit and transform the training data
X = pipeline.fit_transform(merged_df)
y = claim_amount


# If there's output standardization, fit and transform 'ClaimAmount' specifically
if output_standardization:
    y = output_scaler.fit_transform(y.values.reshape(-1, 1))


# Get feature names from the column transformer
feature_names = preprocessor.get_feature_names_out()
# print("New Feature Names:", feature_names)


# Convert sparse Matrice representation of one-hot encoding to full representation
X = X.toarray()
print(X[0], X[0].shape)




#Evaluation
# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import mean_squared_error
# print("Mean Square Error", mean_squared_error(y_pred,y_test))
# print('RMSE',np.sqrt(mean_squared_error(y_pred,y_test)))
# print("Mean Abosulute Error", mean_absolute_error(y_pred,y_test))



def objective(trial, X, y):

    # Suggest values for the hyperparameters
    power_category = trial.suggest_categorical('power_category', ['0', '1-2', '2', '3'])
    if power_category == '1-2':
        power = trial.suggest_float('power_continuous', 1, 2)
    else:
        power = float(power_category)

    alpha = trial.suggest_float('alpha', 1e-3, 10.0, log=True)  # Regularization strength
    solver = trial.suggest_categorical('solver', ['lbfgs', 'newton-cholesky'])
    # Create a model with suggested hyperparameters
    model = TweedieRegressor(power=power, alpha=alpha, solver=solver, max_iter= 300)

    # Perform 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring=make_scorer(mean_squared_error))
    # Compute RMSE from scores
    rmse = (scores.mean()) ** 0.5

    return rmse


def objective_wrapper(trial):
    return objective(trial, X, y)


# Create a study object and optimize the objective function
study = optuna.create_study(direction='minimize')
study.optimize(objective_wrapper, n_trials=2)

# Best trial result
print(f"Best trial: {study.best_trial.value}")
print(f"Best parameters: {study.best_trial.params}")
best_params = study.best_params.copy()

if 'power_continuous' in best_params:
    best_params.pop("power_category")
    best_params["power"] = best_params["power_continuous"]
    best_params.pop("power_continuous")
else:
    best_params["power"] = int(best_params["power_category"])
    best_params.pop("power_category")

model = TweedieRegressor(**best_params)
model.fit(X,y)
print(model.coef_)


#--------------To Do: Visulize the feature weights-------------------
