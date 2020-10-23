import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import json

# File naming and path
data_folder = 'pp_gas_emission'
file_prefix = 'gt_'
file_suffix = '.csv'
dir_sep = '/'


# This is what we need to calculate:
#
# Central tendency: mean, median
# Extremes: percentile 1 and percentile 99
# Dispersion tendency: standard deviation, range
# Linear Interactions: Spearman Rank Correlations (as correlation matrices)

# Let's put that in a structure:
variable_data = {
    'mean': None,
    'median': None,
    'percentile_1': None, # dunno what this is so format might be wrong
    'percentile_99': None,
    'standard_deviation': None,
    'range': None, # or this
    'correlations': None # and this
}

# Function that actually calculates the statistics from the dataframe and puts it in a variable_data object
def calc_stats(df):
    result = {}
    # Group by the columns (the variables)
    for variable in df.columns:
        stats = variable_data.copy()
        stats['mean'] = df[variable].mean()
        stats['median'] = df[variable].median()
        stats['standard_deviation'] = df[variable].std()
        result[variable] = stats
    return result

# Years that we have data from
years = [ '2011', '2012', '2013', '2014', '2015' ]

# Final result object
result_data = {}
# Dataframe that will contain all files combined
total_df = None

for year in years:
    # Read the data of this year
    file_path = data_folder + dir_sep + file_prefix + year + file_suffix
    file_df = pd.read_csv(file_path)
    
    # Add to total dataframe
    if total_df is None:
        total_df = file_df
    else:
        total_df = pd.concat([total_df, file_df])
    
    # Calculate stats and put it in the result under this year
    result_data[year] = calc_stats(file_df)

# Calculate stats of the total dataframe and put it in the result under 'All'
result_data['All'] = calc_stats(total_df)

# Print the resulting object with nice formatting
print(json.dumps(result_data, indent=4))


    


