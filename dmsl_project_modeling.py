import numpy as np
import pandas as pd
import pickle
import shap
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split, ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

pd.set_option('display.width', None)

###############

output_for_modeling_filename = "data/final_data.csv"

############### Querying the dataset

### 1. Load, downsample, and split data

data = pd.read_csv(output_for_modeling_filename)
print(f"Loaded data from file with size: {data.shape}")

# Delete outlier row discovered through 2D PCA plot
outlier_yardmax = 105266.0
data_v2 = data[data["yardage_max"] != outlier_yardmax]

# Drop correlated features
to_drop = ["yardage_min", "yardage_max", "published_year", "style_modern", "construction_bottom_up", "technique_count"]
# to_drop = ["yardage_min", "yardage_max", "published_year", "style_modern", "construction_bottom_up"]
# to_drop = ["yardage_min", "yardage_max", "published_year"]
data_v2 = data_v2.drop(columns=to_drop)

# Downsample class=0 from 90% to 10% to reduce class imbalance
data_v2_class0 = data_v2[ data_v2["popularity_class"] == 0 ]
data_v2_class1 = data_v2[ data_v2["popularity_class"] == 1 ]

data_v2_class0_downsmpl = data_v2_class0.sample(n=10000, random_state=7406)
data_v2_balanced = pd.concat([data_v2_class1, data_v2_class0_downsmpl])

print("\nDownsampled Class 0 (90% of data). Updated class distributions: ")
print(data_v2_balanced["popularity_class"].value_counts().sort_index())

# Split into train, test datasets (80/20) with equal class balance
x = data_v2_balanced.loc[:, data_v2_balanced.columns != "popularity_class"]
y = data_v2_balanced.loc[:, "popularity_class"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, stratify=y, shuffle=True, random_state=7406)
print("\nSize, x_train:", x_train.shape)
print("Size, x_test:", x_test.shape)

# Scaled version of data (e.g. for KNN)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
x_scaled = scaler.transform(x)

### EDA

print("\nEDA:")

# Null and type checks
print(data_v2.info())

# Exploring class shape/separation with PCA
x_pca = PCA(n_components=2).fit_transform(x_scaled)
plt.figure(figsize=(16, 12))
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y)
plt.savefig("data_2d.png")
print("\nPlotting 2d representation of data with PCA: data_2d.png...")
plt.close()

# # Identifying outlier row (from before its deletion)
# outlier_idx = np.argmax(x_pca[:, 0])
# print(f"Outlier row identified through PCA: \n {x.iloc[outlier_idx]} \n class: {y.iloc[outlier_idx]}")

# Focusing PCA plot on bottom left corner
pca_trim = x_pca[:, 1] <= 10
x_pca_trimmed = x_pca[pca_trim]
y_trimmed = y[pca_trim]

plt.figure(figsize=(16, 12))
plt.scatter(x_pca_trimmed[:, 0], x_pca_trimmed[:, 1], c=y_trimmed)
plt.savefig("data_2d_facet.png")
print("\nPlotting bottom-left facet of previous graph: data_2d_facet.png...")
plt.close()

# Exploring multicollinearity - plot created before feature reduction above of correlated features
plt.figure(figsize=(16, 16))
sns.heatmap(x.corr(), annot=True, cmap="coolwarm", fmt=".1f")
plt.savefig("multicollinearity.png")
print("\nPlotting multicollinearity: multicollinearity.png...")
plt.close()

### Data modeling - hypertuning parameters

# Helper function: tune hyperparameters and train model
def tune_train_model(model_dict, scaled=False):
    # Tune hyperparameters
    print("Finding best hyperparameters via GridSearchCV...")
    gs = GridSearchCV(estimator=model_dict["estimator"], param_grid=model_dict["param_grid"], scoring="accuracy", n_jobs=-1)

    if not scaled:
        gs.fit(x_train, y_train)
    else:
        gs.fit(x_train_scaled, y_train)
    model_dict["best_params"] = gs.best_params_

    # Train final model
    model = gs.best_estimator_
    model_dict["best_model"] = model

    # Calculate performance metrics
    if not scaled:
        y_pred = model.predict(x_test)
    else:
        y_pred = model.predict(x_test_scaled)
    classific_report = classification_report(y_test, y_pred, output_dict=True)
    model_dict["results"] = classific_report

# Hyperparameter tuning for all models except KNN
rf_est = RandomForestClassifier(random_state=7406)
param_grid_rf = {
    "n_estimators": [100, 300, 500],            # Number of trees in the forest (default: 100)
    "max_features": ["sqrt", "log2"],           # Number of features randomly sampled at each split (default: sqrt)
    "min_samples_split": [2, 5, 10],            # Minimum samples allowed at a split (default: 2)
    "max_depth": [10, 50, None],                # Max tree depth (default: None)
    "criterion": ["gini", "entropy"]            # Metric used to measure node impurity (default: gini)
}

cb_est = CatBoostClassifier(loss_function="Logloss", eval_metric="Logloss", silent=True, random_seed=7406)
param_grid_cb = {
    "iterations": [500, 1000],                      # Max number of trees (default: 1000)
    "depth": [4, 6, 8],                             # Max tree depth (default: 6)
    "l2_leaf_reg": [3, 5, 10],                      # L2 regularization coefficient (default: 3)
    "random_strength": [0, 1, 5]                    # Randomness added while scoring splits to avoid overfitting (default: 1)
}

xg_est = XGBClassifier(seed=7406)
param_grid_xg = {
    "n_estimators": [100, 300, 500],                # Number of trees (default: 100)
    "learning_rate": [0.01, 0.1, 0.3],              # Step size shrinkage (default: 0.3)
    "max_depth": [4, 6, 8]                          # Max tree depth (default: 6)
}

lr_est = LogisticRegression(random_state=7406)
param_grid_lr1 = {
    "penalty": ["l2"],                              # Regularization - only option for all solver opts (default: l2)
    "C": [0.1, 1.0, 10.0, 100.0],                   # Inverse of regularization strength (default: 1.0)
    "solver": ["lbfgs", "newton-cholesky"]           # Solver type, NW good for big datasets w/ OHEs (default: lbfgs)
}

param_grid_lr2 = {
    "penalty": ["elasticnet"],                      # Regularization(default: l2)
    "l1_ratio": [0.2, 0.5, 0.8],                    # Balance between L1 & L2 in EN (default: 0)
    "C": [0.1, 1.0, 10.0, 100.0],                   # Inverse of regularization strength (default: 1.0)
    "solver": ["saga"],                             # Solver type (default: lbfgs)
    "max_iter": [2000],                             # Max number of iterations (default: 1000)
    "tol": [0.001]                                  # Tolerance (default: 0.0001)
}

nn_est = MLPClassifier(max_iter=500, random_state=7406)
param_grid_nn = {
    "activation": ["logistic", "tanh", "relu"],     # Activation function (default: relu)
    "solver": ["sgd", "adam"],                      # Solver (default: adam)
    "alpha": [0.0001, 0.001, 0.01, 0.1],            # Strength of L2 regularization (default: 0.0001)
    "learning_rate": ["constant", "adaptive"]       # Learning rate schedule (default: constant)
}

models = {
    "RandomForest": {
        "estimator": rf_est,
        "param_grid": param_grid_rf,
        "if_scaled": False
    },
    "CatBoost": {
        "estimator": cb_est,
        "param_grid": param_grid_cb,
        "if_scaled": False
    },
    "XGBoost": {
        "estimator": xg_est,
        "param_grid": param_grid_xg,
    "if_scaled": False
    },
    "LogRegV1": {
        "estimator": lr_est,
        "param_grid": param_grid_lr1,
        "if_scaled": True
    },
    "LogRegV2": {
        "estimator": lr_est,
        "param_grid": param_grid_lr2,
        "if_scaled": True
    },
    "NeuralNet": {
        "estimator": nn_est,
        "param_grid": param_grid_nn,
        "if_scaled": True
    }
}

for model_name, model_dict in models.items():
    print(f"\nTuning model: {model_name}...")

    if model_dict["if_scaled"]:
        tune_train_model(model_dict, scaled=True)
    else:
        tune_train_model(model_dict)

    print("Hyperparameter tuning complete.")
    print(f"Accuracy: {model_dict["results"]["accuracy"]}")
    print(f"Best parameters: {model_dict["best_params"]}")

    with open("pickle/models_dict.pkl", "wb") as f:
        pickle.dump(models, f)

# Hyperparameter tuning for KNN - via elbow plot

print("\nTuning model: KNN...")
test_errs = []
kks = range(1, 27, 2)

for k in kks:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)

    y_pred_test = knn.predict(x_test)
    test_err = 1 - accuracy_score(y_test, y_pred_test)
    test_errs.append(test_err)

plt.figure(figsize=(10,6))
plt.plot(kks, test_errs, label="Test error")
plt.xticks(kks)
plt.xlabel("k")
plt.ylabel("Error rate")
plt.savefig("elbow_plot.png")
print("Saved elbow plot: elbow_plot.png")

# Final KNN model: get accuracy and classification report

best_k = 3
models["KNN"] = {"if_scaled": True}
models["KNN"]["best_params"] = {"k": best_k}

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(x_train_scaled, y_train)
models["KNN"]["best_model"] = knn

y_pred = knn.predict(x_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nFinal model results: \nTest error: {accuracy}")

y_pred = knn.predict(x_test_scaled)
classific_report = classification_report(y_test, y_pred, output_dict=True)
models["KNN"]["results"] = classific_report

with open("pickle/models_dict.pkl", "wb") as f:
    pickle.dump(models, f)

### Data modeling - set up tuned estimators

# with open("pickle/models_dict.pkl", "rb") as file:
#     models = pickle.load(file)

for model_name, model_dict in models.items():
    if model_name != "KNN":
        new_est = clone(model_dict["estimator"])
        model_dict["tuned_estimator"] = new_est.set_params(**model_dict["best_params"])
    else:
        model_dict["tuned_estimator"] = KNeighborsClassifier(n_neighbors=model_dict["best_params"]["k"])

print("\nUpdated all models and saved updated estimators as prep for CV")
with open("pickle/models_dict.pkl", "wb") as f:
    pickle.dump(models, f)

### Data modeling - Monte Carlo CV on each model

def run_cv(models):
    # Stores test errors for all runs
    cv_accs = []

    # Create B=100 random train/test splits
    # Output: iterator with tuple pairs - (train indices, test indices)
    splits = ShuffleSplit(n_splits=100, test_size=0.2, train_size=0.8, random_state=7406).split(data_v2_balanced)

    # Iterate through 100 splits, training each model and storing test error results
    for (train_idx, test_idx) in splits:
        train = data_v2_balanced.iloc[train_idx]
        test = data_v2_balanced.iloc[test_idx]

        x_train, y_train = train.loc[:, data_v2_balanced.columns != "popularity_class"], train.loc[:, "popularity_class"]
        x_test, y_test = test.loc[:, data_v2_balanced.columns != "popularity_class"], test.loc[:, "popularity_class"]

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        temp_accs = []
        for model_name, model_dict in models.items():
            if model_dict["if_scaled"]:
                xtr, xte = x_train_scaled, x_test_scaled
            else:
                xtr, xte = x_train, x_test

            est = model_dict["tuned_estimator"]
            est.fit(xtr, y_train)

            y_pred = est.predict(xte)
            accuracy = accuracy_score(y_test, y_pred)
            temp_accs.append(accuracy)

        cv_accs.append(temp_accs)

    # Aggregate test error metrics for each model
    cv_accs_array = np.array(cv_accs)
    for i, model_name in enumerate(list(models.keys())):
        col = cv_accs_array[:, i]
        models[model_name]["cv_results"] = {
            "mean": np.mean(col),
            "variance": np.var(col),
            "std_dev": np.std(col)
        }

# with open("pickle/models_dict.pkl", "rb") as file:
#     models = pickle.load(file)

print("\nRunning Monte Carlo CV on each model...")
run_cv(models)
print("CV complete")

with open("pickle/models_dict.pkl", "wb") as f:
    pickle.dump(models, f)

# Display results pre- and post-CV

# with open("pickle/models_dict.pkl", "rb") as file:
#     models = pickle.load(file)

summary_data = []
for model_name, model_dict in models.items():
    initial_acc = model_dict["results"]["accuracy"]
    cv_mean = model_dict["cv_results"]["mean"]
    cv_var = model_dict["cv_results"]["variance"]
    cv_std = model_dict["cv_results"]["std_dev"]

    # Append as a dictionary
    summary_data.append({
        "Model": model_name,
        "Initial Accuracy": initial_acc,
        "CV Avg Accuracy": cv_mean,
        "CV Variance": cv_var,
        "CV Std Dev": cv_std
    })

summary_table = pd.DataFrame(summary_data)
print(f"\nPre- and post-CV results: \n {summary_table}")

### Classification report - final Catboost model

# with open("pickle/models_dict.pkl", "rb") as file:
#     models = pickle.load(file)

print("\nClassification report for final CatBoost model...")
print(pd.DataFrame(models["CatBoost"]["results"]).T)

### Feature importance - final Catboost model

# with open("pickle/models_dict.pkl", "rb") as file:
#     models = pickle.load(file)

print("\nCalculating SHAP values for feature importance...")
cb_model = models["CatBoost"]["best_model"]

explainer = shap.TreeExplainer(cb_model)
shap_values = explainer(x_test)

plt.figure(figsize=(16, 12))
shap.summary_plot(shap_values.values, x_test, plot_type="dot", show=False)
plt.tight_layout()
plt.savefig("shap_catboost.png")
plt.close()
print("SHAP beeswarm plot saved: shap_catboost.png")
