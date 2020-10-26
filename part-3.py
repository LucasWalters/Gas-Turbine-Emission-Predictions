import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model

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

Y = training_df['NOX']
X = training_df[input_variable_names]

# Regress on the training data
regr = linear_model.LinearRegression()
regr.fit(X, Y)

# Predict NOX values for the validation data
Ps = regr.predict(validation_df.iloc[:, :-1])

correlation_df = pd.DataFrame({'NOXa': Ps, 'NOXb': validation_df['NOX']}, columns=['NOXa', 'NOXb'])

NOX_correlation = correlation_df.corr(method='spearman')

print(NOX_correlation)
