import pandas as pd
import arff
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, TweedieRegressor
import optuna
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, make_scorer, mean_absolute_error
import seaborn as sns


data_freq = arff.load('freMTPL2freq.arff')
df_freq = pd.DataFrame(data_freq, columns=["IDpol", "ClaimNb", "Exposure", "Area", "VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Density", "Region"])
data_sev = arff.load('freMTPL2sev.arff')
df_sev = pd.DataFrame(data_sev, columns=["IDpol", "ClaimAmount"])
del data_freq, data_sev
# print(np.unique(df_sev["ClaimAmount"])) # The minimum of Claimamount is 1, whereas the maximum is 4e6.
# print(np.unique(df_freq["BonusMalus"], return_counts= True))

# First sum up the "ClaimAmount" with duplicate entries of "IDpol".
# df_sev = df_sev.groupby("IDpol").sum()

df_sev = df_sev.groupby("IDpol", as_index=False).agg({'ClaimAmount': 'sum'})

# There are some nan features in df_freq with some "IDpol" in df_freq, which is directly cleaned,
merged_df = pd.merge(df_freq, df_sev, on="IDpol", how="outer")
merged_df["ClaimAmount"] = merged_df["ClaimAmount"].fillna(0)
merged_df = merged_df.dropna(how="any")
merged_df = merged_df.drop(['IDpol','ClaimNb'], axis = 1)

# Hyper-parameters
random_seed = 35
np.random.seed(random_seed)
input_standardization = True
output_standardization = False
feature_visualization = False


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
    target = merged_df['ClaimAmount'] / merged_df['Exposure']
    plt.figure(figsize=(10, 4))  # Set figure size
    sns.histplot(data=target, kde=False, bins=50)  # KDE for smooth distribution curve
    plt.title(f'Histogram of expected claim amount')
    plt.ylabel('Count')
    plt.xlabel('Expected Claim Amount')
    plt.yscale('log')
    plt.grid(True)
    plt.show()

    # plot_data_distribution(merged_df)
#----------------visualize Features----------------------


# Define the column transformer for handling different types of data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', preprocessing.MinMaxScaler(), [col for col in merged_df.columns if merged_df[col].dtype in [np.float64, np.float32] and col not in ['ClaimAmount', 'Exposure', 'IDpol', 'ClaimNb']]),
        ('cat', preprocessing.OneHotEncoder(handle_unknown='ignore'), [col for col in merged_df.columns if merged_df[col].dtype == 'object']),
    ],
    remainder='drop'
)

# Exclude 'ClaimAmount' and 'Exposure' from any transformation if output_standardization is applied
if output_standardization:
    output_scaler = preprocessing.MinMaxScaler()
claim_amount = merged_df.pop('ClaimAmount') / merged_df.pop('Exposure')  # Separate and transform this column separately

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

# Convert sparse Matrice representation of one-hot encoding to full representation
X = X.toarray()
print(X[0], X[0].shape)

def objective(trial, X, y):
    # Suggest values for the hyperparameters
    power = trial.suggest_float('power', 1, 2)
    alpha = trial.suggest_float('alpha', 1e-3, 10.0, log=True)  # Regularization strength
    solver = trial.suggest_categorical('solver', ['lbfgs', 'newton-cholesky'])
    # Create a model with suggested hyperparameters
    model = TweedieRegressor(power=power, alpha=alpha, solver=solver, max_iter= 500)
    # Perform 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
    scores = cross_val_score(model, X, y, cv=kf, scoring=make_scorer(mean_absolute_error))
    # Compute RMSE from scores
    # rmse = (scores.mean()) ** 0.5
    abs_error = scores.mean()

    return abs_error

def objective_wrapper(trial):
    return objective(trial, X, y)

# Create a study object and optimize the objective function
study = optuna.create_study(direction='minimize')
study.optimize(objective_wrapper, n_trials=10)

# Best trial result
print(f"Best trial: {study.best_trial.value}")
print(f"Best parameters: {study.best_trial.params}")
best_params = study.best_params.copy()

model = TweedieRegressor(**best_params)
model.fit(X,y)


#--------------To Do: Visualize the feature weights-------------------
# Get feature names from the column transformer
feature_names = preprocessor.get_feature_names_out()
# print("New Feature Names:", feature_names)

# Plot feature importance
plt.figure()
# plt.bar(range(len(feature_importances)), feature_importances, alpha=0.7)

# Step to categorize and average gradients
feature_groups = {}
for name, weight in zip(feature_names, model.coef_):
    # Parse the feature name to find the base feature (before one-hot encoding)
    if '__' in name:
        base_name = name.split('__')[1].split('_')[0]
        if base_name not in feature_groups:
            feature_groups[base_name] = []
        feature_groups[base_name].append(abs(weight))

# Calculating the mean gradient for each original feature
mean_weights = {feature: np.mean(weight)  for feature, weight in feature_groups.items()}

# Sorting for better visualization
sorted_features = sorted(mean_weights, key=mean_weights.get, reverse=True)
sorted_importances = [mean_weights[feature] for feature in sorted_features]


plt.barh(sorted_features, sorted_importances, color='skyblue')
plt.ylabel('Feature Index')
plt.xlabel('Importance')
plt.title('Feature Importance from Tweedie Regression')
plt.show()