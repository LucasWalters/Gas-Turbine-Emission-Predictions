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

### Predict NOX values for the validation data
train_data = training_df[input_variable_names]
train_obs = training_df['NOX']

# Predict on validation data
val_data = validation_df.iloc[:, :-1]
val_obs = validation_df['NOX']

apply_linear_regression(train_data, train_obs, val_data, val_obs)


### Predict NOX values for the test data
train_data = pd.concat([training_df, validation_df])[input_variable_names] 
train_obs = pd.concat([training_df, validation_df])['NOX']

# Predict on test data
test_data = test_df.iloc[:, :-1]
test_obs = test_df['NOX']

apply_linear_regression(train_data, train_obs, test_data, test_obs)


if apply_pca:
    # Scale the data on unit scale (mean = 0, variance = 1)
    scaler = StandardScaler()
    # Fit on training set only.
    scaler.fit(train_data)
    # Apply transform to both the training set and the test set.
    train_scaled = scaler.transform(train_data)
    test_scaled = scaler.transform(test_df[input_variable_names])
    # Make an instance of the Model. .95 means the minimum number of principal components such that 95% of the variance is retained.
    pca = PCA(.95)
    
    pca.fit(train_scaled)
    train_pca = pca.transform(train_scaled)
    test_pca = pca.transform(test_scaled)
    
    train_scaled_df = pd.DataFrame(train_scaled, columns = [input_variable_names])
    train_pca_df = pd.DataFrame(train_pca, columns = ['pca_' + str(x) for x in range(len(train_pca[0]))])
    
    test_scaled_df = pd.DataFrame(test_scaled, columns = [input_variable_names])
    test_pca_df = pd.DataFrame(test_pca, columns = ['pca_' + str(x) for x in range(len(train_pca[0]))])
    
    train_merged = pd.concat([train_scaled_df, train_pca_df], axis = 1)
    test_merged = pd.concat([test_scaled_df, test_pca_df], axis = 1)
    
    apply_linear_regression(train_merged, train_obs, test_merged, test_df['NOX'])

