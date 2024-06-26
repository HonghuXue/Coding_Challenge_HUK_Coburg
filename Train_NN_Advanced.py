
# ----To add:  feature outlier / D square-----
import seaborn as sns
import pandas as pd
import arff
import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_tweedie_deviance
import matplotlib.pyplot as plt

# Set random seed for reproducibility
random_seed = 35
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# Load data
data_freq = arff.load('freMTPL2freq.arff')
df_freq = pd.DataFrame(data_freq, columns=["IDpol", "ClaimNb", "Exposure", "Area", "VehPower", "VehAge", "DrivAge", "BonusMalus",
                                "VehBrand", "VehGas", "Density", "Region"])
data_sev = arff.load('freMTPL2sev.arff')
df_sev = pd.DataFrame(data_sev, columns=["IDpol", "ClaimAmount"])
del data_freq, data_sev

# Preprocess and merge data
df_sev = df_sev.groupby("IDpol", as_index=False).agg({'ClaimAmount': 'sum'})
merged_df = pd.merge(df_freq, df_sev, on="IDpol", how="outer")
merged_df["ClaimAmount"] = merged_df["ClaimAmount"].fillna(0)
merged_df = merged_df.dropna(how="any")
merged_df = merged_df.drop(['IDpol'], axis = 1)


# Hyper-parameters
feature_visualization = False
show_evaluation_plot = True
input_standardization = True
output_standardization = True
include_sampling_weights = True
tweedie_loss_train = True
tweedie_loss_val = False
if tweedie_loss_val:
    validation_criterion = 'TweedieLoss'  # 'MSELoss' 'TweedieLoss'
else:
    validation_criterion = 'L1Loss'  # 'MSELoss' 'TweedieLoss'
if tweedie_loss_train or tweedie_loss_val:
    rho = 1.8
batch_size = 4096
num_epochs = 500
K_fold_splits = 5

# for early stopping criteria

epochs_no_improve = 0  # Counter for epochs without improvement
n_epochs_stop = 30  # Number of epochs to stop after no improvement

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
            if column != "ClaimAmount":
                sns.histplot(data=df, x=column, kde=False, bins=30)  # KDE for smooth distribution curve
                plt.title(f'Histogram of {column}')
            elif column == "ClaimAmount":
                sns.histplot(data=df[column]/df['Exposure'], kde=False, bins=30)
                plt.title(f'Histogram of expected {column} per year')

        plt.ylabel('Count')
        plt.xlabel(column)
        plt.grid(True)
        plt.show()


if feature_visualization:
    plot_data_distribution(merged_df)
#----------------visualize Features----------------------



class CustomNetwork(nn.Module):
    def __init__(self, layer_sizes=(46, 128, 128, 128, 1)):
        super(CustomNetwork, self).__init__()

        # Ensure the layer_sizes is a tuple or list
        assert isinstance(layer_sizes, (tuple, list)), "layer_sizes must be a tuple or a list"
        assert len(layer_sizes) > 2, "layer_sizes must include input, at least one hidden layer, and output dimensions"

        # Create a list to hold layers
        layers = []

        # Iterate over the list of sizes to create the layers
        for i in range(len(layer_sizes) - 1):
            # Append the linear layer based on current and next layer sizes
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

            # Append BatchNorm layer except for the output layer
            if i < len(layer_sizes) - 2:
                layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
                # layers.append(nn.PReLU())
                layers.append(nn.GELU())
                layers.append(nn.Dropout(p=0.5))

        # Convert the list of layers into nn.Sequential
        if tweedie_loss_train or tweedie_loss_val:
            layers.append(nn.Softplus())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # Pass the input through the sequential layers
        x = self.layers(x)
        return x



# Prepare data for cross-validation
kf = KFold(n_splits=K_fold_splits, shuffle=True, random_state=random_seed)
# Metrics for evaluation
metrics = {
    'D_square': lambda y_true, y_pred: 1 - mean_squared_error(y_true, y_pred) / np.var(y_true),
    'R_square': r2_score,
    'MAE': mean_absolute_error,
    'RMSE': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))
}

# Feature encoding
feature_transformer = ColumnTransformer(transformers=[
    ('num', MinMaxScaler(), [col for col in merged_df.columns if
                             merged_df[col].dtype in [np.float64, np.float32] and col not in ['ClaimAmount', 'Exposure', 'IDpol', 'ClaimNb']]),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
     [col for col in merged_df.columns if merged_df[col].dtype == 'object'])
], remainder='passthrough')

feature_transformer.fit_transform(merged_df.drop(['ClaimAmount', 'Exposure', 'ClaimNb'], axis=1))
feature_names = feature_transformer.get_feature_names_out()
print("New Feature Names:", feature_names)

device = "cuda" if torch.cuda.is_available() else "cpu"
if include_sampling_weights:
    weights = torch.tensor(np.array(merged_df["Exposure"]), dtype=torch.float32, device=device)
else:
    weights = torch.tensor(np.ones(merged_df["Exposure"].shape), dtype=torch.float32, device=device)


# -----outlier detection-----
# merged_df['ClaimRatio'] = merged_df['ClaimAmount'] / merged_df['Exposure']
# Q1 = merged_df['ClaimRatio'].quantile(0.25)
# Q3 = merged_df['ClaimRatio'].quantile(0.75)
# IQR = Q3 - Q1
# print(Q3, Q1)
# # Define the upper and lower bounds for outliers
# lower_bound = Q1 - 1.5 * IQR
# upper_bound = Q3 + 1.5 * IQR
#
# # Cap values exceeding the upper bound
# merged_df['CappedClaimRatio'] = merged_df['ClaimRatio'].apply(lambda x: min(x, upper_bound))
#
# # Display some statistics or the head of the dataframe to verify
# print(merged_df[['ClaimRatio', 'CappedClaimRatio']].describe())
# merged_df.drop('ClaimRatio', axis=1, inplace=True)


# Main loop for cross-validation
results = {key: np.empty((K_fold_splits, num_epochs)) * np.nan for key in metrics.keys()}

k_fold_iter = 0
for train_index, test_index in kf.split(merged_df):
    train_df, test_df = merged_df.iloc[train_index], merged_df.iloc[test_index]
    X_train, y_train = feature_transformer.fit_transform(train_df.drop(['ClaimAmount', 'Exposure', 'ClaimNb'], axis=1)), train_df['ClaimAmount']/train_df['Exposure']
    X_test, y_test = feature_transformer.transform(test_df.drop(['ClaimAmount', 'Exposure'], axis=1)), test_df['ClaimAmount']/test_df['Exposure']

    if output_standardization:
        output_scaler = MinMaxScaler(feature_range=(0, 1))
        y_train = output_scaler.fit_transform(y_train.values.reshape(-1,1))
        y_test = output_scaler.transform(y_test.values.reshape(-1,1))
    y_train_min = y_train.min()
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test, dtype=torch.float32, device=device)

    # The implementation of only 2 weights.
    # weights_minor_class = len(train_df["ClaimAmount"]) / (2 * np.sum(train_df["ClaimAmount"] > 0))
    # weights_major_class = len(train_df["ClaimAmount"]) / (2 * np.sum(train_df["ClaimAmount"] == 0))
    # print("weights_for_class_imbalance", weights_minor_class, weights_major_class)

    # Train a model
    model = CustomNetwork(layer_sizes=(X_train.shape[1], 128, 128, 128, 1)).to(device)
    model_backup = CustomNetwork(layer_sizes=(X_train.shape[1], 128, 128, 128, 1)).to(device)
    model_backup.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Define scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.995)
    if validation_criterion == "L1Loss":
        criterion = nn.L1Loss(reduction = 'none')
    elif validation_criterion == "MSELoss":
        criterion = nn.MSELoss(reduction='none')

    # ... include the training loop here ...
    # Lists to store losses for plotting
    train_losses, train_losses_unscaled = [] , []
    val_losses, val_losses_unscaled = [] , []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()

        batch_train = torch.randperm(X_train.shape[0])
        batch_train = batch_train[(batch_train.numel() % batch_size):]
        batch_train = batch_train.view(-1, batch_size)

        running_loss, running_loss_unscaled = 0, 0

        # for each mini-batch
        for i in range(batch_train.size(0)):

            optimizer.zero_grad()
            input = X_train[batch_train[i], :]
            target = y_train[batch_train[i]]
            predictions = model(input)
            if tweedie_loss_train:
                loss = -target * torch.pow(predictions, 1 - rho) / (1 - rho) + torch.pow(predictions, 2 - rho) / (2 - rho)
                loss_original_scale = -target.detach().cpu().numpy() * (1/output_scaler.scale_) * np.power(predictions.detach().cpu().numpy() * (1/output_scaler.scale_), 1 - rho) / (1 - rho) + np.power(predictions.detach().cpu().numpy() * (1/output_scaler.scale_), 2 - rho) / (2 - rho)
            else:
                loss = criterion(predictions, target)

            # Add weights to address class imbalance
            if include_sampling_weights:
                # weights = torch.where(target == y_train_min, weights_major_class, weights_minor_class)  # Increase the weight for non-zero targets
                loss = (loss * weights[batch_train[i]].unsqueeze(-1)).mean()  # Weighted loss
                loss_original_scale = (loss_original_scale * weights[batch_train[i]].unsqueeze(-1).detach().cpu().numpy()).mean()
            else:
                loss = loss.mean()
                loss_original_scale = loss_original_scale.mean()
            loss.backward()
            optimizer.step()
            # statistics
            running_loss += loss.item()
            running_loss_unscaled += loss_original_scale

        running_loss /= batch_train.size(0)
        running_loss_unscaled /= batch_train.size(0)

        if output_standardization and not tweedie_loss_train:
            train_losses.append(running_loss * 1 / output_scaler.scale_)  # Record training loss
        else:
            train_losses.append(running_loss)  # Record training loss
            train_losses_unscaled.append(np.squeeze(running_loss_unscaled))
        scheduler.step()  # Update learning rate


        # ----Validation----
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_test)
            if validation_criterion == 'MSELoss' or validation_criterion == 'L1Loss':
                val_loss = criterion(val_predictions, y_test)
            elif validation_criterion == 'TweedieLoss':
                val_loss = -y_test * torch.pow(val_predictions, 1 - rho) / (1 - rho) + torch.pow(val_predictions, 2 - rho) / (2 - rho)
                val_loss_original_scale = -y_test.detach().cpu().numpy() * (1 / output_scaler.scale_) * np.power(
                    val_predictions.detach().cpu().numpy() * (1 / output_scaler.scale_), 1 - rho) / (1 - rho) + np.power(
                    val_predictions.detach().cpu().numpy() * (1 / output_scaler.scale_), 2 - rho) / (2 - rho)

            if include_sampling_weights:
                val_loss = val_loss * weights[test_index].unsqueeze(-1)
                if validation_criterion == 'TweedieLoss':
                    val_loss_original_scale = val_loss_original_scale * weights[test_index].unsqueeze(-1).detach().cpu().numpy()
            val_loss = val_loss.mean()
            if validation_criterion == 'TweedieLoss':
                val_loss_original_scale = val_loss_original_scale.mean()

        if output_standardization and not tweedie_loss_val:
            val_losses.append(val_loss.item() * 1 / output_scaler.scale_)
        else:
            val_losses.append(val_loss.item())
            val_losses_unscaled.append(val_loss_original_scale)
        current_lr = scheduler.get_last_lr()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, LR: {current_lr[0]:.6f}')

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            model_backup.load_state_dict(model.state_dict())
        else:
            epochs_no_improve += 1

        # Evaluation for different matrix
        val_predictions = val_predictions.cpu().numpy()
        y_test_np = y_test.detach().cpu().numpy()
        for name, metric_fn in metrics.items():
            results[name][k_fold_iter][epoch] = metric_fn(y_test_np * 1/output_scaler.scale_, val_predictions * 1/output_scaler.scale_)
            print(name, results[name][k_fold_iter][epoch])
        if epochs_no_improve == n_epochs_stop:
            print('Early stopping triggered')
            model.load_state_dict(model_backup.state_dict())
            del model_backup
            break

    if show_evaluation_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses_unscaled, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss ' + validation_criterion)
        for name, metric_fn in metrics.items():
            plt.plot(results[name][k_fold_iter], label='Validation Loss ' + name)
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        plt.show()

    k_fold_iter += 1
    print('K_fold_iter', k_fold_iter)


# Evaluation metrics results
# for name, scores in results.items():
#     print(f'{name}: {np.nanmean(scores):.4f} ± {np.nanstd(scores):.4f}')

# ... continue with the rest of the code for visualization and further analysis ...
#---------------------------------------Visualize Feature Importance---------------------------------------
# Gradient-based Feature Importance (for Neural Networks)
# For neural networks, gradients can be used as a measure of feature importance. This is similar to sensitivity analysis but involves taking the gradient of the output with respect to each input feature.
#
# This method and sensitivity analysis are particularly useful for deep learning models where internal weights and their relationship with features are often opaque and not linear.


def gradient_based_feature_importance(model, input_data, weights, feature_names, output_scaler):
    """
    Calculate gradient-based feature importance for a PyTorch model.

    Parameters:
    - model: A trained PyTorch model.
    - input_data: A batch of input data (torch.Tensor).

    Returns:
    - feature_importances: An array of feature importances.
    """
    # Ensure the model is in evaluation mode
    model.eval()

    # Enable gradient calculation with respect to the input
    input_data.requires_grad = True

    # Perform a forward pass
    outputs = model(input_data)

    # Compute gradients
    outputs.backward(torch.ones_like(outputs))

    # Extract the gradients of the output with respect to inputs
    gradients = (input_data.grad.abs() * weights.unsqueeze(-1)).mean(dim=0)

    # Detach gradients and convert to numpy for further analysis/plotting
    feature_importances = gradients.detach().cpu().numpy()

    # Plot feature importance
    plt.figure()
    # plot all the features
    # plt.bar(range(len(feature_importances)), feature_importances, alpha=0.7)

    # Step to categorize and average gradients
    feature_groups = {}
    for name, grad in zip(feature_names, feature_importances):
        # Parse the feature name to find the base feature (before one-hot encoding)
        if '__' in name:
            base_name = name.split('__')[1].split('_')[0]
            if base_name not in feature_groups:
                feature_groups[base_name] = []
            feature_groups[base_name].append(grad)

    # Calculating the mean gradient for each original feature
    mean_gradients = {feature: np.mean(grads * 1 / output_scaler.scale_)  for feature, grads in feature_groups.items()}

    # Sorting for better visualization
    sorted_features = sorted(mean_gradients, key=mean_gradients.get, reverse=True)
    sorted_importances = [mean_gradients[feature] for feature in sorted_features]


    plt.barh(sorted_features, sorted_importances, color='skyblue')
    plt.ylabel('Feature Index')
    plt.xlabel('Importance')
    plt.title('Gradient-based Feature Importance')
    plt.show()

    return feature_importances

# Compute the importance
importance = gradient_based_feature_importance(model, X_train, weights[train_index], feature_names, output_scaler)
