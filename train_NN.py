# To add:
# Evaluation matrix ()  D square
# cross validation
# importance sampling ratio
# Tweedie loss (Done in training, but also in validation)


import pandas as pd
import arff
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

random_seed = 35
torch.manual_seed(random_seed)
np.random.seed(random_seed)

data_freq = arff.load('freMTPL2freq.arff')
df_freq = pd.DataFrame(data_freq, columns=["IDpol", "ClaimNb", "Exposure", "Area", "VehPower", "VehAge","DrivAge", "BonusMalus", "VehBrand", "VehGas", "Density", "Region"])
data_sev = arff.load('freMTPL2sev.arff')
df_sev = pd.DataFrame(data_sev, columns=["IDpol", "ClaimAmount"])
del data_freq, data_sev
# print(np.unique(df_sev["ClaimAmount"])) # The minimum of Claimamount is 1, whereas the maximum is 4e6.

# Hyper-parameters
merge_by_intersection = False
input_standardization = True
output_standardization = True
include_sampling_weights = False
tweedie_loss = True
if tweedie_loss:
    rho = 1.6
batch_size = 4096
num_epochs = 500

# for early stopping criteria
best_val_loss = float('inf')  # Initialize best validation loss to a very high value
epochs_no_improve = 0  # Counter for epochs without improvement
n_epochs_stop = 30  # Number of epochs to stop after no improvement

# original length of df_sev["IDpol"] is 26639 . After groupby 24950.   After merge: 24944
# After checking, dev_freq["CLaimNB"] is almost the same as same my defined statistics , except for one entry with the difference of 1 out of 24944 entries
# df_sev["claimNB_sev"] = np.ones(df_sev["IDpol"].shape)
# print("CHECK : {}".format(np.unique(merged_df["claimNB_sev"]-merged_df["ClaimNb"], return_counts=True)))



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
    # n_samples / (n_classes * np.bincount(y)), where n_class = 2
    # weights_minor_class = np.sum(merged_df["ClaimAmount"] == 0)/ np.sum(merged_df["ClaimAmount"] > 0)
    weights_minor_class = len(merged_df["ClaimAmount"]) / (2 * np.sum(merged_df["ClaimAmount"] > 0))
    weights_major_class = len(merged_df["ClaimAmount"]) / (2 * np.sum(merged_df["ClaimAmount"] == 0))
    print("weights_for_class_imbalance", weights_minor_class, weights_major_class)

# There are some nan features in df_freq with some "IDpol" in df_freq, which is directly cleaned,
merged_df = merged_df.drop(['IDpol','ClaimNb'], axis = 1)

# ------To Do: Feature value visulization-------

# input_scalers_pred_claimamount = []
# output_scalers_pred_claimamount = []

# Split the data into training and test sets
train_df, test_df = train_test_split(merged_df, test_size=0.2, random_state=random_seed)


# Define the column transformer for handling different types of data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', preprocessing.MinMaxScaler(), [col for col in train_df.columns if train_df[col].dtype in [np.float64, np.float32] and col not in ['ClaimAmount', 'Exposure', 'IDpol', 'ClaimNb']]),
        ('cat', preprocessing.OneHotEncoder(handle_unknown='ignore'), [col for col in train_df.columns if train_df[col].dtype == 'object']),
    ],
    remainder='passthrough'
)

# Exclude 'ClaimAmount' and 'Exposure' from any transformation if output_standardization is applied
if output_standardization:
    output_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
claim_amount = train_df.pop('ClaimAmount') / train_df.pop('Exposure')  # Separate and transform this column separately

# Create a processing pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    # ('output_scaler', output_scaler) if output_standardization else ('dummy', 'passthrough')
])

# Fit and transform the training data
X_train = pipeline.fit_transform(train_df)
y_train = claim_amount

# Transform the test data using the same pipeline
X_test = pipeline.transform(test_df)
y_test = test_df['ClaimAmount'] / test_df['Exposure']


# If there's output standardization, fit and transform 'ClaimAmount' specifically
if output_standardization:
    y_train = output_scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test = output_scaler.transform(test_df['ClaimAmount'].values.reshape(-1, 1) / test_df['Exposure'].values.reshape(-1, 1))

y_train_min = y_train.min()
# # Example to save scalers and encoder
# input_scalers_pred_claimamount.append(('num', preprocessor.named_transformers_['num']))
# input_scalers_pred_claimamount.append(('cat', preprocessor.named_transformers_['cat']))
# if output_standardization:
#     output_scalers_pred_claimamount.append(output_scaler)

# Get feature names from the column transformer
feature_names = preprocessor.get_feature_names_out()
print("New Feature Names:", feature_names)


# Convert sparse Matrix representation of one-hot encoding to full representation
X_train, X_test = X_train.toarray(), X_test.toarray()
print(X_train[0], X_train[0].shape)
# print(X_train.max(axis=1), X_train.min(axis=1), X_train.std(axis=1) )
# print(X_test.max(axis=1), X_test.min(axis=1), X_test.std(axis=1) )



class CustomNetwork(nn.Module):
    def __init__(self, layer_sizes=(4, 128, 128, 128, 1)):
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
        if tweedie_loss:
            layers.append(nn.Softplus())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # Pass the input through the sequential layers
        x = self.layers(x)

        return x


device = "cuda" if torch.cuda.is_available() else "cpu"

num_recordings_train = X_train.shape[0]

model = CustomNetwork(layer_sizes=(X_train.shape[1], 64, 128, 256 ,1)).to(device)
model_backup = CustomNetwork(layer_sizes=(X_train.shape[1], 64, 128, 256 ,1)).to(device)
model_backup.load_state_dict(model.state_dict())

# print(model)

X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
y_train = torch.tensor(y_train, dtype=torch.float32, device=device)#.unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32, device=device)#.unsqueeze(1)

# Loss function and optimizer
criterion = nn.L1Loss(reduction = 'none')
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define scheduler
scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.995)

# ----------------Training loop-----------------

# Lists to store losses for plotting
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()

    batch_train = torch.randperm(num_recordings_train)
    batch_train = batch_train[(batch_train.numel() % batch_size):]
    batch_train = batch_train.view(-1, batch_size)

    running_loss = 0

    # for each mini-batch
    for i in range(batch_train.size(0)):
        # mini-batch
        optimizer.zero_grad()
        input = X_train[batch_train[i], :]
        target = y_train[batch_train[i]]
        predictions = model(input)
        if tweedie_loss:
            loss = -target * torch.pow(predictions, 1 - rho) / (1 - rho) + torch.pow(predictions, 2 - rho) / (2 - rho)
            # print(torch.pow(predictions, 1 - rho) / (1 - rho))
            # print(target, predictions, torch.pow(predictions, 1 - rho) , torch.pow(predictions, 2 - rho) )
        else:
            loss = criterion(predictions, target)

        # Add weights to address class imbalance
        if include_sampling_weights:
            weights = torch.where(target == y_train_min, weights_major_class, weights_minor_class)  # Increase the weight for non-zero targets
            loss = (loss * weights).mean()  # Weighted loss
        else:
            loss = loss.mean()
        loss.backward()
        optimizer.step()
        # statistics
        running_loss += loss.item()

    running_loss /= batch_train.size(0)

    if output_standardization and not tweedie_loss:
        train_losses.append(running_loss * 1 / output_scaler.scale_)  # Record training loss
    else:
        train_losses.append(running_loss)  # Record training loss
    scheduler.step()  # Update learning rate

    # Validation
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_test)
        val_loss = criterion(val_predictions, y_test)
        # weights = torch.where(y_test == y_train_min, weights_major_class, weights_minor_class)  # Increase the weight for non-zero targets
        # val_loss = (val_loss * weights).mean()
        val_loss = val_loss.mean()
    if output_standardization and not tweedie_loss:
            val_losses.append(val_loss.item() * 1 / output_scaler.scale_)
    else:
        val_losses.append(val_loss.item() )


    current_lr = scheduler.get_last_lr()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, LR: {current_lr[0]:.6f}')

    # Check for early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        model_backup.load_state_dict(model.state_dict())
    else:
        epochs_no_improve += 1

    if epochs_no_improve == n_epochs_stop:
        print('Early stopping triggered')
        model.load_state_dict(model_backup.state_dict())
        del model_backup
        break


# Plotting the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.show()



#---------------------------------------Visualize Feature Importance---------------------------------------
# Gradient-based Feature Importance (for Neural Networks)
# For neural networks, gradients can be used as a measure of feature importance. This is similar to sensitivity analysis but involves taking the gradient of the output with respect to each input feature.
#
# This method and sensitivity analysis are particularly useful for deep learning models where internal weights and their relationship with features are often opaque and not linear.


def gradient_based_feature_importance(model, input_data, feature_names, output_scaler):
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

    # Assuming outputs are scalar (e.g., regression task)
    # If not, you might need to adjust how the backward pass is handled
    # For example, sum up outputs if they are more than one per sample:
    # outputs = outputs.sum()

    # Compute gradients
    outputs.backward(torch.ones_like(outputs))

    # Extract the gradients of the output with respect to inputs
    gradients = input_data.grad.abs().mean(dim=0)

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
importance = gradient_based_feature_importance(model, X_train, feature_names, output_scaler)
