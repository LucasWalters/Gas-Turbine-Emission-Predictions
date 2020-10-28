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
    train_scaled = pd.DataFrame(scaler.transform(training_df[input_variable_names]), columns = input_variable_names)
    val_scaled = pd.DataFrame(scaler.transform(validation_df[input_variable_names]), columns = input_variable_names)
    test_scaled = pd.DataFrame(scaler.transform(test_df[input_variable_names]), columns = input_variable_names)
    return (train_scaled, training_df['NOX'], val_scaled, validation_df['NOX'], test_scaled, test_df['NOX'])

def compute_performance(name, pred, observed):
    correlation_df = pd.DataFrame({'NOXa': pred, 'NOXb': observed}, columns=['NOXa', 'NOXb'])
    NOX_correlation = correlation_df.corr(method='spearman').iloc[1][0]
    NOX_mae = mean_absolute_error(observed, pred)
    NOX_r2 = correlation_df.corr(method='pearson').iloc[1][0] ** 2

    print(name + " NOX Spearman Correlation: " + str(NOX_correlation))
    print(name + " NOX Mean absolute error: " + str(NOX_mae))
    print(name + " NOX R^2: " + str(NOX_r2))


def apply_linear_regression(training_data, training_target, test_data, test_target, prefix = "[TEST]"):
    # Regress on the training data
    regr = linear_model.LinearRegression()
    regr.fit(training_data, training_target)

    # Predict on test data
    test_pred = regr.predict(test_data)

    compute_performance(prefix, test_pred, test_target)

def phase1(training_df, training_t, validation_df, validation_t, test_df, test_t):
    ### Predict NOX values for the validation data
    apply_linear_regression(training_df, training_t, validation_df, validation_t, "[VAL]")

    ### Predict NOX values for the test data
    X = pd.concat([training_df, validation_df])
    Y = pd.concat([training_t, validation_t])

    apply_linear_regression(X, Y, test_df, test_t, "[TEST]")

def phase2(training_df, training_t, validation_df, validation_t, test_df, test_t):
    train_reduced = training_df[reduced_input_variable_names]
    test_reduced = test_df[reduced_input_variable_names]
    validation_reduced = validation_df[reduced_input_variable_names]
    
    apply_linear_regression(train_reduced, training_t, validation_reduced, validation_t, "[VAL-REDUCED]")

    if apply_pca:
        # Make an instance of the Model. .95 means the minimum number of principal components such that 95% of the variance is retained.
        pca = PCA(.95)
        
        pca.fit(train_reduced)
        
        train_pca = pca.transform(train_reduced)
        test_pca = pca.transform(test_reduced)
        validation_pca = pca.transform(validation_reduced)
        
        train_pca_df = pd.DataFrame(train_pca, columns = ['pca_' + str(x) for x in range(len(train_pca[0]))])
        test_pca_df = pd.DataFrame(test_pca, columns = ['pca_' + str(x) for x in range(len(test_pca[0]))])
        validation_pca_df = pd.DataFrame(validation_pca, columns = ['pca_' + str(x) for x in range(len(validation_pca[0]))])
        
        train_merged = pd.concat([train_reduced, train_pca_df], axis = 1)
        test_merged = pd.concat([test_reduced, test_pca_df], axis = 1)
        validation_merged = pd.concat([validation_reduced, validation_pca_df], axis = 1)
        
        apply_linear_regression(train_merged, training_t, validation_merged, validation_t, "[VAL-PCA]")
        apply_linear_regression(train_merged, training_t, test_merged, test_t, "[TEST-PCA]")

        

(training_df, training_t, validation_df, validation_t, test_df, test_t) = z_normalize(training_df, validation_df, test_df)
phase1(training_df, training_t, validation_df, validation_t, test_df, test_t)
phase2(training_df, training_t, validation_df, validation_t, test_df, test_t)

