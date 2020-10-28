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

input_variable_names = ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP']
reduced_input_variable_names = ['AT', 'AP', 'AFDP', 'TIT', 'TAT', 'TEY', 'CDP']

# Years that we have data from
years = [ '2011', '2012', '2013', '2014', '2015' ]

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

target_column = 'NOX'
# Normalize and split the data
(train_data, train_out) = z_normalize_and_seperate_target(train_df, input_variable_names, target_column)
(val_data, val_out) = z_normalize_and_seperate_target(val_df, input_variable_names, target_column)
(test_data, test_out) = z_normalize_and_seperate_target(test_df, input_variable_names, target_column)

def apply_linear_regression(training_data, training_target, test_data, test_target):
    # Regress on the training data
    regr = linear_model.LinearRegression()
    regr.fit(training_data, training_target)

    # Predict on test data
    test_pred = regr.predict(test_data)
    
    (SC, MAE, R2) = compute_performance(test_pred, test_target)
    print_performance(SC, MAE, R2)
    return pd.DataFrame(test_pred, columns=[target_column])

def phase1(training_df, training_t, validation_df, validation_t, test_df, test_t):
    ### Predict NOX values for the validation data
    print("PREDICTION ON VALIDATION DATA")
    apply_linear_regression(training_df, training_t, validation_df, validation_t)

    ### Predict NOX values for the test data
    X = pd.concat([training_df, validation_df])
    Y = pd.concat([training_t, validation_t])

    #apply_linear_regression(X, Y, test_df, test_t)

def phase2(train_data, train_out, val_data, val_out, test_data, test_out):
    #find_features(reduced_input_variable_names, train_data.copy(), train_out, val_data.copy(), val_out)
    
    train_reduced = train_data[reduced_input_variable_names]
    val_reduced = val_data[reduced_input_variable_names]
    test_reduced = test_data[reduced_input_variable_names]
    print("On validation with reduced input")
    apply_linear_regression(train_reduced, train_out, val_reduced, val_out)

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
        
        
        print("On validation with PCA")
        apply_linear_regression(train_merged, train_out, val_merged, val_out)
        print("On test with PCA")
        apply_linear_regression(train_merged, train_out, test_merged, test_out)

def predict_blocks(block_train_data, blocks, variable_columns, target_columns):
    print("Testing block", 10-len(blocks))
    result = apply_linear_regression(block_train_data[variable_columns], block_train_data[target_columns], blocks[0][variable_columns], blocks[0][target_columns]) 
    blocks[0][target_columns] = result
    if len(blocks) > 1:
        predict_blocks(pd.concat([block_train_data, blocks[0]]), np.delete(blocks, 0), variable_columns, target_columns)



def phase3(train_data, data_2013, data_2014, data_2015, variable_names, target_columns):
    val_blocks = np.array_split(data_2013, 10)
    val_blocks = [df.reset_index(drop = True) for df in val_blocks]
    test_blocks = np.append(np.array_split(data_2014, 10), np.array_split(data_2015, 10))
    test_blocks = [df.reset_index(drop = True) for df in test_blocks]
    predict_blocks(train_data.reset_index(drop = True), val_blocks, variable_names, target_columns)
    

#phase1(training_df, training_t, validation_df, validation_t, test_df, test_t)
# phase2(train_data, train_out, val_data, val_out, test_data, test_out)


phase3(train_df, total_df['2013'], total_df['2014'], total_df['2015'], input_variable_names, target_column)





