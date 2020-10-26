import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import json
import seaborn as sns


print_data = False
print_corr_matrix = True
save_csv = False
save_corr_matrices = False
save_bar_chart = False


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
    'percentile_1': None,
    'percentile_99': None,
    'standard_deviation': None,
    'range': None
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
    return result

# Years that we have data from
years = [ '2011', '2012', '2013', '2014', '2015' ]

# Final result object
result_data = {}
# Object that will contain the correlation matrices
correlation_matrices = {}
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
    # Calculate correlation matrix and put it under this year
    correlation_matrices[year] = file_df.corr(method='spearman')
    


# Calculate stats of the total dataframe and put it in the result under 'All'
result_data['All'] = calc_stats(total_df)
correlation_matrices['All'] = total_df.corr(method='spearman')

if print_data:
    # Print the resulting object with nice formatting
    print(json.dumps(result_data, indent=4))

if print_corr_matrix:
    # Print correlation matrix of 2011
    print(correlation_matrices['2011'])

# Let's turn that into a multi indexed dataframe, should've probably done this at the start but only thought of it now so whatever
# Get first index array - the years and 'All'
years_and_all = np.array(list(result_data.keys()))
# Get second index array - the variable names (AT, AP, AFDP, etc)
variables = np.array(list(result_data['All'].keys()))

statistics = np.array(list(variable_data.keys()))
# Make multi index
midx = pd.MultiIndex.from_product([years_and_all, variables])

# Create the actual dataframe, the columns are the statistics names (mean, median, etc)
df = pd.DataFrame(index = midx, columns = statistics)

# Fill dataframe with data we got earlier
for index in years_and_all:
    for variable in result_data[index]:
        for statistic in result_data[index][variable]:
            value = result_data[index][variable][statistic]
            df[statistic][index][variable] = value

if print_data:
    # Print the start of it, we can now easily use this for the plots
    print(df.head())

if save_csv:
    # Write yearly data to file
    for variable in variable_data.keys():
        df[variable] = df[variable].astype(float)
    df.to_csv('data.csv', float_format='%.2f')

if save_corr_matrices:
# Write correlation matrices to file and plot them
    for index in correlation_matrices:
        correlation_matrices[index].to_csv('correlation_'+str(index)+'.csv', float_format='%.2f')

        plt.figure(figsize=(8, 5.5))
        sns.heatmap(correlation_matrices[index].round(2), annot=True, vmin=-1, vmax=1, center=0, cmap='coolwarm', square=True)
        plt.yticks(rotation=0) 
        plt.savefig('Correlations_'+str(index)+'.png',bbox_inches='tight')



if save_bar_chart:
    # Prepare data for bar charts
    df_swapped = df.swaplevel(0, 1, axis=0)
    labels = years_and_all
    fig, ax = plt.subplots()
    x = []  # the label locations
    width = 0.1  # the width of the bars
    label_width = width * len(variables)
    for i in range(len(labels)):
        x.append(label_width * i)
    x = np.array(x)
    print(x)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


    for i, variable in enumerate(variables):
        variable_means = list(df_swapped['mean'][variable])
        rects = ax.bar(x + i * width, variable_means, width, label=variable, align='edge')
        autolabel(rects)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Mean')
    ax.set_title('Mean of variables by year')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    plt.show()
    


