import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import json
import seaborn as sns

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
    'range': None
    #'correlations': None # and this
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
        stats['range'] = np.ptp(df[variable])
        stats['percentile_1'] = np.percentile(df[variable], 1)
        stats['percentile_99'] = np.percentile(df[variable], 99)
        result[variable] = stats
    # Just kind of awkwardly stick this matrix in the results
    result['correlation'] = df.corr(method='spearman')
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

# Let's turn that into a multi indexed dataframe, should've probably done this at the start but only thought of it now so whatever
# Get first index array - the years and 'All'
years_and_all = np.array(list(result_data.keys()))
# Get second index array - the variable names (AT, AP, AFDP, etc)
variables = np.array(list(result_data['All'].keys()))
# Make multi index
midx = pd.MultiIndex.from_product([years_and_all, variables])

# Create the actual dataframe, the columns are the statistics names (mean, median, etc)
df = pd.DataFrame(index = midx, columns = list(variable_data.keys()))

# Fill dataframe with data we got earlier
for year in result_data:
    for variable in result_data[year]:
        if(variable == 'correlation'): continue
        for statistic in result_data[year][variable]:
            value = result_data[year][variable][statistic]
            df[statistic][year][variable] = value

# Print the start of it, we can now easily use this for the plots
print(df.head())

# Write yearly data to file
for variable in variable_data.keys():
    df[variable] = df[variable].astype(float)
df.to_csv('data.csv', float_format='%.2f')

# Write correlation matrices to file
for year in years:
    result_data[year]['correlation'].to_csv('correlation_'+str(year)+'.csv', float_format='%.2f')
    sns.heatmap(result_data[year]['correlation'].round(2), annot=True, vmin=-1, vmax=1, center=0, cmap='coolwarm', square=True)
    plt.show()

# TODO charts:

# labels = result_data['All'].keys()



# labels = ['G1', 'G2', 'G3', 'G4', 'G5']
# men_means = [20, 34, 30, 35, 27]
# women_means = [25, 32, 34, 20, 25]

# x = np.arange(len(labels))  # the label locations
# width = 0.35  # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, men_means, width, label='Men')
# rects2 = ax.bar(x + width/2, women_means, width, label='Women')

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Scores')
# ax.set_title('Scores by group and gender')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend()


# def autolabel(rects):
    # """Attach a text label above each bar in *rects*, displaying its height."""
    # for rect in rects:
        # height = rect.get_height()
        # ax.annotate('{}'.format(height),
                    # xy=(rect.get_x() + rect.get_width() / 2, height),
                    # xytext=(0, 3),  # 3 points vertical offset
                    # textcoords="offset points",
                    # ha='center', va='bottom')


# autolabel(rects1)
# autolabel(rects2)

# fig.tight_layout()

# plt.show()

    


