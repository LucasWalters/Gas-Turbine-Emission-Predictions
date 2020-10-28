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

# Split the data into input and output data
train_data = train_df[input_variable_names]
train_out = train_df['NOX']
val_data = val_df[input_variable_names]
val_out = val_df['NOX']
test_data = test_df[input_variable_names]
test_out = test_df['NOX']

# Normalize the data
(train_data, val_data, test_data) = z_normalize(train_data, val_data, test_data, input_variable_names)

target_column = 'NOX'

def apply_linear_regression(training_data, training_target, test_data, test_target):
    # Regress on the training data
    regr = linear_model.LinearRegression()
    regr.fit(training_data, training_target)

    # Predict on test data
    test_pred = regr.predict(test_data)
    
    (SC, MAE, R2) = compute_performance(test_pred, test_target)
    print_performance(SC, MAE, R2)
    return ((SC, MAE, R2), pd.DataFrame(test_pred, columns=[target_column]))

def compute_val_and_test_performance(train_data, train_out, val_data, val_out, test_data, test_out):
    ### Predict NOX values for the validation data
    print("> Validation performance")
    (vSC, vMAE, vR2) = apply_linear_regression(train_data, train_out, val_data, val_out)[0]

    ### Predict NOX values for the test data
    X = pd.concat([train_data, val_data])
    Y = pd.concat([train_out, val_out])

    print("> Test performance")
    (tSC, tMAE, tR2) = apply_linear_regression(X, Y, test_data, test_out)[0]
    return ((vSC, vMAE, vR2), (tSC, tMAE, tR2))

def phase2(baseline, train_data, train_out, val_data, val_out, test_data, test_out):
    #feature_comb = find_features(baseline[0], input_variable_names, train_data.copy(), train_out, val_data.copy(), val_out)
    feature_comb = [('AH', 'TAT'), ('AFDP', 'TAT'), ('GTEP', 'TIT'), ('TEY', 'CDP')]
    
    train_data = apply_engineered_features(train_data, feature_comb)
    val_data = apply_engineered_features(val_data, feature_comb)
    test_data = apply_engineered_features(test_data, feature_comb)

    print("Engineered validation performance")
    apply_linear_regression(train_data, train_out, val_data, val_out, "[VAL-ENGINEERED]")
    print("Engineered test performance")
    train_val_data = pd.concat([train_data, val_data])
    train_val_out = pd.concat([train_out, val_out])
    apply_linear_regression(train_val_data, train_val_out, test_data, test_out, "[TEST-ENGINEERED]")

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
    #return (pd.concat([block_train_data, blocks[0]])    
        
def predict_blocks2(block_train_data, blocks, variable_columns, target_columns):
    print("Testing block", 20-len(blocks))
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
    #predict_blocks2(predicted_val_data.reset_index(drop = True), test_blocks, variable_names, target_columns)

### Phase 1: Compute validation and test performance for original features
print(">>> Original Feature Performance")
baseline = compute_val_and_test_performance(train_data, train_out, val_data, val_out, test_data, test_out)

### Find new engineered features
#feature_comb = find_features(baseline[0], input_variable_names, train_data.copy(), train_out, val_data.copy(), val_out)
feature_comb = [('AH', 'TAT'), ('AFDP', 'TAT'), ('GTEP', 'TIT'), ('TEY', 'CDP')]

### Apply features to all datasets
train_data = apply_engineered_features(train_data, feature_comb)
val_data = apply_engineered_features(val_data, feature_comb)
test_data = apply_engineered_features(test_data, feature_comb)

### Phase 2: Compute validation and test performance for engineered features
print(">>> Engineered Feature Performance")
compute_val_and_test_performance(train_data, train_out, val_data, val_out, test_data, test_out)

### Phase 3
phase3(train_df, total_df['2013'], total_df['2014'], total_df['2015'], input_variable_names, target_column)
