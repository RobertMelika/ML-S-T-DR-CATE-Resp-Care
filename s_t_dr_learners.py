import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, GradientBoostingClassifier, \
    RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from econml.metalearners import SLearner
from econml.metalearners import TLearner
from econml.dr import LinearDRLearner, DRLearner
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from nb21 import cumulative_gain, elast
from lightgbm import LGBMRegressor
from datasets import sim1, sim2, sim3, sim4, sim5, sim6
from catenets.models.jax import SNet, TNet, DRNet


def learnGB(simulation, samples):
    d, ate, X, W, Y, TE = simulation(samples)

    # X is the features, W is the treatment indicator, Y is the observed outcome, TE is the actual treatment effect
    # Generate a DataFrame from the simulated data
    data = pd.DataFrame(X, columns=[f"feature_{i + 1}" for i in range(d)])
    data['treatment'] = W
    data['outcome'] = Y
    data['real_te'] = TE

    # Define treatment and outcome columns
    treatment_column = 'treatment'
    outcome_column = 'outcome'
    real_te_column = 'real_te'

    # Define features (excluding treatment, outcome and real treatment effect)
    features = list(set(data.columns) - {treatment_column, outcome_column, real_te_column})

    # Confounding variables
    confounders = []

    # Train test split
    x_train, x_test, y_train, y_test, t_train, t_test, TE_train, TE_test = train_test_split(
        data[features], data[outcome_column], data[treatment_column], data[real_te_column], test_size=0.2,
        random_state=768)

    TE_train_arr = TE[x_train.index]
    TE_test_arr = TE[x_test.index]

    # S-Learner with Random Forest and Gradient boosting
    s_learner = SLearner(overall_model=GradientBoostingRegressor(n_estimators=100, random_state=768))
    s_learner.fit(Y=y_train.astype(int), T=t_train, X=x_train)

    # Estimate CATE for each learner
    cate_s_learner = s_learner.effect(x_test)
    mse_s_learner = mean_squared_error(TE_test, cate_s_learner)

    # Print the mean and standard deviation for CATE estimates from each learner
    print("Samples:", samples)
    print("S-Learner - Mean CATE:", np.mean(cate_s_learner), "MSE:", mse_s_learner, "ATE:", ate)

    # T-Learner with Gradient boosting
    t_learner = TLearner(models=GradientBoostingRegressor(n_estimators=100, random_state=768))
    t_learner.fit(y_train.astype(int), X=x_train, T=t_train)

    cate_t_learner = t_learner.effect(x_test)
    mse_t_learner = mean_squared_error(TE_test, cate_t_learner)

    print("T-Learner - Mean CATE:", np.mean(cate_t_learner), "MSE:", mse_t_learner)

    # DR-Learner with Gradient Boosting for regression and Gradient boosting for propensity

    dr_learner = LinearDRLearner(
        model_propensity=GradientBoostingClassifier(n_estimators=100, random_state=768),
        model_regression=GradientBoostingRegressor(n_estimators=100, random_state=768),
        discrete_outcome=False
    )

    features_without_confounders = list(set(features) - set(confounders))

    # print(features_without_confounders)

    # dr_learner.fit(y_train.astype(int), t_train, X=X_train[features_without_confounders], W=X_train[confounders])
    dr_learner.fit(y_train.astype(int), t_train, X=x_train[features_without_confounders])

    cate_dr_learner = dr_learner.effect(x_test[features_without_confounders])
    mse_dr_learner = mean_squared_error(TE_test, cate_dr_learner)

    print("DR-Learner - Mean CATE:", np.mean(cate_dr_learner), "MSE:", mse_dr_learner)

    return mse_s_learner, mse_t_learner, mse_dr_learner

def learnLinear(simulation, samples):
    d, ate, X, W, Y, TE = simulation(samples)

    # X is the features, W is the treatment indicator, Y is the observed outcome, TE is the actual treatment effect
    # Generate a DataFrame from the simulated data
    data = pd.DataFrame(X, columns=[f"feature_{i + 1}" for i in range(d)])
    data['treatment'] = W
    data['outcome'] = Y
    data['real_te'] = TE

    # Define treatment and outcome columns
    treatment_column = 'treatment'
    outcome_column = 'outcome'
    real_te_column = 'real_te'

    # Define features (excluding treatment, outcome and real treatment effect)
    features = list(set(data.columns) - {treatment_column, outcome_column, real_te_column})

    # Confounding variables
    confounders = []

    # Train test split
    x_train, x_test, y_train, y_test, t_train, t_test, TE_train, TE_test = train_test_split(
        data[features], data[outcome_column], data[treatment_column], data[real_te_column], test_size=0.2,
        random_state=768)

    TE_train_arr = TE[x_train.index]
    TE_test_arr = TE[x_test.index]

    # S-Learner with Random Forest and Gradient boosting
    s_learner = SLearner(overall_model=LinearRegression())
    s_learner.fit(Y=y_train.astype(int), T=t_train, X=x_train)

    # Estimate CATE for each learner
    cate_s_learner = s_learner.effect(x_test)
    mse_s_learner = mean_squared_error(TE_test, cate_s_learner)

    # Print the mean and standard deviation for CATE estimates from each learner
    print("Samples:", samples)
    print("S-Learner - Mean CATE:", np.mean(cate_s_learner), "MSE:", mse_s_learner, "ATE:", ate)

    # T-Learner with Gradient boosting
    t_learner = TLearner(models=LinearRegression())
    t_learner.fit(y_train.astype(int), X=x_train, T=t_train)

    cate_t_learner = t_learner.effect(x_test)
    mse_t_learner = mean_squared_error(TE_test, cate_t_learner)

    print("T-Learner - Mean CATE:", np.mean(cate_t_learner), "MSE:", mse_t_learner)

    # DR-Learner with Gradient Boosting for regression and Gradient boosting for propensity

    dr_learner = DRLearner(
        model_propensity=LogisticRegression(random_state=768),
        model_regression=LinearRegression(),
        discrete_outcome=False
    )
    # dr_learner = DRLearner()

    features_without_confounders = list(set(features) - set(confounders))

    # print(features_without_confounders)

    # dr_learner.fit(y_train.astype(int), t_train, X=X_train[features_without_confounders], W=X_train[confounders])
    dr_learner.fit(y_train.astype(int), t_train, X=x_train[features_without_confounders])

    cate_dr_learner = dr_learner.effect(x_test[features_without_confounders])
    mse_dr_learner = mean_squared_error(TE_test, cate_dr_learner)

    print("DR-Learner - Mean CATE:", np.mean(cate_dr_learner), "MSE:", mse_dr_learner)

    return mse_s_learner, mse_t_learner, mse_dr_learner

def learnNeuralNetwork(simulation, samples):
    d, ate, X, W, Y, TE = simulation(samples)

    # X is the features, W is the treatment indicator, Y is the observed outcome, TE is the actual treatment effect
    # Generate a DataFrame from the simulated data
    data = pd.DataFrame(X, columns=[f"feature_{i + 1}" for i in range(d)])
    data['treatment'] = W
    data['outcome'] = Y
    data['real_te'] = TE

    # Define treatment and outcome columns
    treatment_column = 'treatment'
    outcome_column = 'outcome'
    real_te_column = 'real_te'

    # Define features (excluding treatment, outcome and real treatment effect)
    features = list(set(data.columns) - {treatment_column, outcome_column, real_te_column})

    # Confounding variables
    confounders = []

    # Train test split
    x_train, x_test, y_train, y_test, t_train, t_test, TE_train, TE_test = train_test_split(
        data[features], data[outcome_column], data[treatment_column], data[real_te_column], test_size=0.2,
        random_state=768)

    TE_train_arr = TE[x_train.index]
    TE_test_arr = TE[x_test.index]

    # y_train = y_train.values.reshape((y_train.shape[0], 1))
    # Convert y_train to numpy array to avoid the reshape error
    y_train = y_train.values.reshape(-1, 1)  # Convert to numpy array and reshape to (n_obs, 1)
    y_test = y_test.values.reshape(-1, 1)  # Convert to numpy array and reshape to (n_obs, 1)

    # ensure t_train and t_test are numpy arrays
    t_train = t_train.values
    t_test = t_test.values

    # shape_y = y.shape
    #      21     if len(shape_y) == 1:
    #      22         # should be shape (n_obs, 1), not (n_obs,)
    # ---> 23         return y.reshape((shape_y[0], 1))
    #      24     return y

    # S-Learner with NeuralNetworks
    s_learner = SNet()
    s_learner.fit(y=y_train, w=t_train, X=x_train)

    # Estimate CATE for each learner
    cate_s_learner = s_learner.predict(x_test)
    mse_s_learner = mean_squared_error(TE_test, cate_s_learner)

    # Print the mean and standard deviation for CATE estimates from each learner
    print("Samples:", samples)
    print("S-Learner - Mean CATE:", np.mean(cate_s_learner), "MSE:", mse_s_learner, "ATE:", ate)

    # T-Learner with Neural Networks
    t_learner = TNet()
    t_learner.fit(y=y_train, w=t_train, X=x_train)

    cate_t_learner = t_learner.predict(x_test)
    mse_t_learner = mean_squared_error(TE_test, cate_t_learner)

    print("T-Learner - Mean CATE:", np.mean(cate_t_learner), "MSE:", mse_t_learner)

    # DR-Learner with Neural Networks

    dr_learner = DRNet()

    features_without_confounders = list(set(features) - set(confounders))

    # print(features_without_confounders)

    # dr_learner.fit(y_train.astype(int), t_train, X=X_train[features_without_confounders], W=X_train[confounders])
    dr_learner.fit(y=y_train, w=t_train, X=x_train)

    cate_dr_learner = dr_learner.predict(x_test[features_without_confounders])
    mse_dr_learner = mean_squared_error(TE_test, cate_dr_learner)

    print("DR-Learner - Mean CATE:", np.mean(cate_dr_learner), "MSE:", mse_dr_learner)

    return mse_s_learner, mse_t_learner, mse_dr_learner


def learnLinearBinary(simulation, samples):
    d, ate, X, W, Y, TE = simulation(samples)

    # X is the features, W is the treatment indicator, Y is the observed outcome, TE is the actual treatment effect
    # Generate a DataFrame from the simulated data
    data = pd.DataFrame(X, columns=[f"feature_{i + 1}" for i in range(d)])
    data['treatment'] = W
    data['outcome'] = Y
    data['real_te'] = TE

    # Define treatment and outcome columns
    treatment_column = 'treatment'
    outcome_column = 'outcome'
    real_te_column = 'real_te'

    # Define features (excluding treatment, outcome and real treatment effect)
    features = list(set(data.columns) - {treatment_column, outcome_column, real_te_column})

    # Confounding variables
    confounders = []

    # Train test split
    x_train, x_test, y_train, y_test, t_train, t_test, TE_train, TE_test = train_test_split(
        data[features], data[outcome_column], data[treatment_column], data[real_te_column], test_size=0.2,
        random_state=768)

    TE_train_arr = TE[x_train.index]
    TE_test_arr = TE[x_test.index]

    # S-Learner with Random Forest and Gradient boosting
    s_learner = SLearner(overall_model=LinearRegression())
    s_learner.fit(Y=y_train.astype(int), T=t_train, X=x_train)

    # Estimate CATE for each learner
    cate_s_learner = s_learner.effect(x_test)
    mse_s_learner = mean_squared_error(TE_test, cate_s_learner)

    # Print the mean and standard deviation for CATE estimates from each learner
    print("Samples:", samples)
    print("S-Learner - Mean CATE:", np.mean(cate_s_learner), "MSE:", mse_s_learner, "ATE:", ate)

    # T-Learner with Gradient boosting
    t_learner = TLearner(models=LinearRegression())
    t_learner.fit(y_train.astype(int), X=x_train, T=t_train)

    cate_t_learner = t_learner.effect(x_test)
    mse_t_learner = mean_squared_error(TE_test, cate_t_learner)

    print("T-Learner - Mean CATE:", np.mean(cate_t_learner), "MSE:", mse_t_learner)

    # DR-Learner with Gradient Boosting for regression and Gradient boosting for propensity

    dr_learner = DRLearner(
        model_propensity=LogisticRegression(random_state=768),
        model_regression=LogisticRegression(random_state=768),
        model_final=LinearRegression(),
        discrete_outcome=True
    )
    # dr_learner = DRLearner()

    features_without_confounders = list(set(features) - set(confounders))

    # print(features_without_confounders)

    # dr_learner.fit(y_train.astype(int), t_train, X=X_train[features_without_confounders], W=X_train[confounders])
    dr_learner.fit(y_train.astype(int), t_train, X=x_train[features_without_confounders])

    cate_dr_learner = dr_learner.effect(x_test[features_without_confounders])
    mse_dr_learner = mean_squared_error(TE_test, cate_dr_learner)

    print("DR-Learner - Mean CATE:", np.mean(cate_dr_learner), "MSE:", mse_dr_learner)

    return mse_s_learner, mse_t_learner, mse_dr_learner

def learnGBBinary(simulation, samples):
    d, ate, X, W, Y, TE = simulation(samples)

    # X is the features, W is the treatment indicator, Y is the observed outcome, TE is the actual treatment effect
    # Generate a DataFrame from the simulated data
    data = pd.DataFrame(X, columns=[f"feature_{i + 1}" for i in range(d)])
    data['treatment'] = W
    data['outcome'] = Y
    data['real_te'] = TE

    # Define treatment and outcome columns
    treatment_column = 'treatment'
    outcome_column = 'outcome'
    real_te_column = 'real_te'

    # Define features (excluding treatment, outcome and real treatment effect)
    features = list(set(data.columns) - {treatment_column, outcome_column, real_te_column})

    # Confounding variables
    confounders = []

    # Train test split
    x_train, x_test, y_train, y_test, t_train, t_test, TE_train, TE_test = train_test_split(
        data[features], data[outcome_column], data[treatment_column], data[real_te_column], test_size=0.2,
        random_state=768)

    TE_train_arr = TE[x_train.index]
    TE_test_arr = TE[x_test.index]

    # S-Learner with Random Forest and Gradient boosting
    s_learner = SLearner(overall_model=GradientBoostingRegressor(n_estimators=100, random_state=768))
    s_learner.fit(Y=y_train.astype(int), T=t_train, X=x_train)

    # Estimate CATE for each learner
    cate_s_learner = s_learner.effect(x_test)
    mse_s_learner = mean_squared_error(TE_test, cate_s_learner)

    # Print the mean and standard deviation for CATE estimates from each learner
    print("Samples:", samples)
    print("S-Learner - Mean CATE:", np.mean(cate_s_learner), "MSE:", mse_s_learner, "ATE:", ate)

    # T-Learner with Gradient boosting
    t_learner = TLearner(models=GradientBoostingRegressor(n_estimators=100, random_state=768))
    t_learner.fit(y_train.astype(int), X=x_train, T=t_train)

    cate_t_learner = t_learner.effect(x_test)
    mse_t_learner = mean_squared_error(TE_test, cate_t_learner)

    print("T-Learner - Mean CATE:", np.mean(cate_t_learner), "MSE:", mse_t_learner)

    # DR-Learner with Gradient Boosting for regression and Gradient boosting for propensity

    dr_learner = LinearDRLearner(
        model_propensity=GradientBoostingClassifier(n_estimators=100, random_state=768),
        model_regression=GradientBoostingClassifier(n_estimators=100, random_state=768),
        discrete_outcome=True
    )

    features_without_confounders = list(set(features) - set(confounders))

    # print(features_without_confounders)

    # dr_learner.fit(y_train.astype(int), t_train, X=X_train[features_without_confounders], W=X_train[confounders])
    dr_learner.fit(y_train.astype(int), t_train, X=x_train[features_without_confounders])

    cate_dr_learner = dr_learner.effect(x_test[features_without_confounders])
    mse_dr_learner = mean_squared_error(TE_test, cate_dr_learner)

    print("DR-Learner - Mean CATE:", np.mean(cate_dr_learner), "MSE:", mse_dr_learner)

    return mse_s_learner, mse_t_learner, mse_dr_learner