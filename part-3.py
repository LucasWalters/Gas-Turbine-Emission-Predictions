import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

apply_pca = True

# File naming and path
data_folder = 'pp_gas_emission'
file_prefix = 'gt_'
file_suffix = '.csv'
dir_sep = '/'

# Let's put that in a structure:
variable_data = {
    'mean': None,
    'median': None,
    'percentile_1': None,
    'percentile_99': None,
    'standard_deviation': None,
    'range': None
}

input_variable_names = ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP']
reduced_input_variable_names = ['AT', 'AP', 'AFDP', 'TIT', 'TAT', 'TEY', 'CDP']

# Years that we have data from
years = [ '2011', '2012', '2013', '2014', '2015' ]

# Final result object
result_data = {}
correlation_matrices = {}
# Dataframe that will contain all files combined
total_df = {}

for year in years:
    # Read the data of this year
    file_path = data_folder + dir_sep + file_prefix + year + file_suffix
    file_df = pd.read_csv(file_path)
    del file_df['CO']

    # Add to total dataframe
    total_df[year] = file_df

training_df = pd.concat([total_df['2011'], total_df['2012']])
validation_df = total_df['2013']
test_df = pd.concat([total_df['2014'], total_df['2015']])

def z_normalize(training_df, validation_df, test_df):
    # Scale the data on unit scale (mean = 0, variance = 1)
    scaler = StandardScaler()
    # Fit on training set only.
    scaler.fit(training_df[input_variable_names])
    # Apply transform to both the training set and the test set.
    train_scaled = scaler.transform(training_df[input_variable_names])
    val_scaled = scaler.transform(validation_df[input_variable_names])
    test_scaled = scaler.transform(test_df[input_variable_names])
    return (train_scaled, val_scaled, test_scaled)

z_normalize(training_df, validation_df, test_df)


def compute_performance(name, pred, observed):
    correlation_df = pd.DataFrame({'NOXa': pred, 'NOXb': observed}, columns=['NOXa', 'NOXb'])
    NOX_correlation = correlation_df.corr(method='spearman').iloc[1][0]
    NOX_mae = mean_absolute_error(observed, pred)
    NOX_r2 = correlation_df.corr(method='pearson').iloc[1][0] ** 2

    print(name + " NOX Spearman Correlation: " + str(NOX_correlation))
    print(name + " NOX Mean absolute error: " + str(NOX_mae))
    print(name + " NOX R^2: " + str(NOX_r2))


def apply_linear_regression(training_data, training_target, test_data, test_target):
    # Regress on the training data
    regr = linear_model.LinearRegression()
    regr.fit(training_data, training_target)

    # Predict on test data
    test_pred = regr.predict(test_data)

    compute_performance("[TEST]", test_pred, test_target)

def phase1(training_df, validation_df, test_df):
    ### Predict NOX values for the validation data
    Y = training_df['NOX']
    X = training_df[input_variable_names]

    # Regress on the training data
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)

    # Predict on validation data
    val_pred = regr.predict(validation_df[input_variable_names])
    val_obs = validation_df['NOX']

    compute_performance("[VAL]", val_pred, val_obs)

    ### Predict NOX values for the test data
    Y = pd.concat([training_df, validation_df])['NOX']
    X = pd.concat([training_df, validation_df])[input_variable_names]

    # Regress on the training data
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)

    # Predict on test data
    test_pred = regr.predict(test_df.iloc[:, :-1])
    test_obs = test_df['NOX']

    compute_performance("[TEST]", test_pred, test_obs)

def phase2(training_df, validation_df, test_df):
    ### Predict NOX values for the validation data
    Y = training_df['NOX']
    X = training_df[reduced_input_variable_names]

    # Regress on the training data
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)

    # Predict on validation data
    val_pred = regr.predict(validation_df[reduced_input_variable_names])
    val_obs = validation_df['NOX']

    if apply_pca:
        # Make an instance of the Model. .95 means the minimum number of principal components such that 95% of the variance is retained.
        pca = PCA(.95)
        
        pca.fit(training_df)
        pca_training_data = pca.transform(training_df)
        pca_test_data = pca.transform(test_df)
        
        print(pd.DataFrame(pca_training_data).head())
        
        apply_linear_regression(pca_training_data, training_df['NOX'], pca_test_data, test_df['NOX'])

        compute_performance("[ENG-VAL]", val_pred, val_obs)

phase1(training_df, validation_df, test_df)
phase2(training_df, validation_df, test_df)

