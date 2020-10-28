import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils import *

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

train_df = pd.concat([total_df['2011'], total_df['2012']])
val_df = total_df['2013']
test_df = pd.concat([total_df['2014'], total_df['2015']])

# Split the data into input and output data
train_data = train_df[input_variable_names]
train_out = train_df['NOX']
val_data = val_df[input_variable_names]
val_out = val_df['NOX']
test_data = test_df[input_variable_names]
test_out = test_df['NOX']

# Normalize the data
(train_data, val_data, test_data) = z_normalize(train_data, val_data, test_data, input_variable_names)

def apply_linear_regression(training_data, training_target, test_data, test_target, prefix = "[TEST]"):
    # Regress on the training data
    regr = linear_model.LinearRegression()
    regr.fit(training_data, training_target)

    # Predict on test data
    test_pred = regr.predict(test_data)
    
    (SC, MAE, R2) = compute_performance(prefix, test_pred, test_target)
    print_performance(SC, MAE, R2)

def phase1(training_df, training_t, validation_df, validation_t, test_df, test_t):
    ### Predict NOX values for the validation data
    apply_linear_regression(training_df, training_t, validation_df, validation_t, "[VAL]")

    ### Predict NOX values for the test data
    X = pd.concat([training_df, validation_df])
    Y = pd.concat([training_t, validation_t])

    #apply_linear_regression(X, Y, test_df, test_t, "[TEST]")

def phase2(train_data, train_out, val_data, val_out, test_data, test_out):
    #find_features(reduced_input_variable_names, train_data.copy(), train_out, val_data.copy(), val_out)
    
    train_reduced = train_data[reduced_input_variable_names]
    val_reduced = val_data[reduced_input_variable_names]
    test_reduced = test_data[reduced_input_variable_names]

    apply_linear_regression(train_reduced, train_out, val_reduced, val_out, "[VAL-REDUCED]")

    if apply_pca:
        # Make an instance of the Model. .95 means the minimum number of principal components such that 95% of the variance is retained.
        pca = PCA(.95)
        
        pca.fit(train_reduced)
        
        train_pca = pca.transform(train_reduced)
        val_pca = pca.transform(val_reduced)
        test_pca = pca.transform(test_reduced)
        
        train_pca_df = pd.DataFrame(train_pca, columns = ['pca_' + str(x) for x in range(len(train_pca[0]))])
        val_pca_df = pd.DataFrame(val_pca, columns = ['pca_' + str(x) for x in range(len(val_pca[0]))])
        test_pca_df = pd.DataFrame(test_pca, columns = ['pca_' + str(x) for x in range(len(test_pca[0]))])
        
        train_merged = pd.concat([train_reduced, train_pca_df], axis = 1)
        val_merged = pd.concat([val_reduced, val_pca_df], axis = 1)
        test_merged = pd.concat([test_reduced, test_pca_df], axis = 1)
        
        apply_linear_regression(train_merged, train_out, val_merged, val_out, "[VAL-PCA]")
        apply_linear_regression(train_merged, train_out, test_merged, test_out, "[TEST-PCA]")

#phase1(training_df, training_t, validation_df, validation_t, test_df, test_t)
phase2(train_data, train_out, val_data, val_out, test_data, test_out)
