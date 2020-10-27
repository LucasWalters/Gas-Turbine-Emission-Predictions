import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import json
import seaborn as sns
import scipy.stats as sc


print_data = False
print_corr_matrix = False
save_csv = False
save_corr_matrices = False
save_bar_chart = False
save_line_chart = False
do_t_test = True


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
p_df = None
p_matrices = {}
# Dataframe that will contain all files combined
total_df = None

for year in years:
    # Read the data of this year
    file_path = data_folder + dir_sep + file_prefix + year + file_suffix
    file_df = pd.read_csv(file_path)
    del file_df['CO']

    # Add to total dataframe
    if total_df is None:
        total_df = file_df
    else:
        total_df = pd.concat([total_df, file_df])

    # if do_t_test:
    #     print(file_df['NOX'])
    #     variables = np.array(list(file_df.columns[:-2]))
    #     for variable in variables:
    #         stat, p = ttest_rel(list(file_df[variable]), list(file_df['NOX']))
    #         print('T test ', year, variable, stat, p)

    # Calculate stats and put it in the result under this year
    result_data[year] = calc_stats(file_df)
    # Calculate correlation matrix and put it under this year
    variables = np.array(list(file_df.columns))
    matrix, p = sc.spearmanr(file_df)
    matrix = pd.DataFrame(matrix, index=variables, columns=variables)
    p = pd.DataFrame(p, index=variables, columns=variables)
    if p_df is None:
        p_df = pd.DataFrame(index=variables)
    p_df[year] = p['NOX']
    correlation_matrices[year] = matrix
    p_matrices[year] = p



# Calculate stats of the total dataframe and put it in the result under 'All'
result_data['All'] = calc_stats(total_df)
variables = np.array(list(result_data['All'].keys()))
matrix, p = sc.spearmanr(total_df)
matrix = pd.DataFrame(matrix, index=variables, columns=variables)
p = pd.DataFrame(p, index=variables, columns=variables)
p_df['All'] = p['NOX']
correlation_matrices['All'] = matrix
p_matrices['All'] = p
# for index in p_matrices:
#     print('\n',index, ':')
#     print(p_matrices[index]['NOX'])

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

corr_change_variables = variables[:-1]

if do_t_test:
    # stat, p = ttest_rel(data1, data2)

    # data1 = 5 * np.random.randn(100) + 50
    # data2 = 5 * randn(100) + 51
    df_swapped = df.swaplevel(0, 1, axis=0)

    for variable in corr_change_variables:
        stat, p = sc.ttest_ind(list(df_swapped['mean'][variable][:-1]), list(df_swapped['mean']['NOX'][:-1]))
        # print('T test ', variable, stat, p)
    #
    # for variable in corr_change_variables:
    #     variable_corrs = []
    #     for index in years:
    #         variable_corrs.append(correlation_matrices[index]['NOX'][variable])
    #     # print(variable_corrs)

if save_corr_matrices:
    # Write correlation matrices to file and plot them
    for index in correlation_matrices:
        correlation_matrices[index].to_csv('correlation_'+str(index)+'.csv', float_format='%.2f')

        plt.figure(figsize=(8, 5.5))
        sns.heatmap(correlation_matrices[index].round(2), annot=True, vmin=-1, vmax=1, center=0, cmap='coolwarm', square=True)
        plt.yticks(rotation=0)
        plt.savefig('Correlations_'+str(index)+'.png',bbox_inches='tight')

        # Render correlation change matrix
        change_matrix = abs(correlation_matrices['2015'] - correlation_matrices['2011']).round(2)
        plt.figure(figsize=(8, 5.5))

        colors = [(0, '#eeeeee'),
                  (0.2, '#eeeeee'),
                  (0.20001, 'lightgrey'),
                  (0.7, '#b30326'),
                  (1, '#b30326')]

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('help', colors)
        mask = np.triu(np.ones_like(change_matrix, dtype=bool))
        sns.heatmap(change_matrix, annot=True, mask=mask, vmin=0.0, vmax=1.0, center=0.5, cmap=cmap, square=True)
        plt.yticks(rotation=0)
        plt.savefig('Correlations_Change.png',bbox_inches='tight')


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
    label_width = width * len(corr_change_variables)
    for i in range(len(labels)):
        x.append(label_width * i)
    x = np.array(x)



    for i, variable in enumerate(corr_change_variables):
        variable_corrs = []
        for index in years_and_all:
            variable_corrs.append(correlation_matrices[index]['NOX'][variable])

        rects = ax.bar(x + i * width, variable_corrs, width, label=variable, align='edge')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Correlation with NOx')
    ax.set_title('Correlation with NOx of variables by year')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    plt.show()

if save_line_chart:
    # Prepare data for bar charts
    df_swapped = df.swaplevel(0, 1, axis=0)
    labels = years
    fig, ax = plt.subplots()
    # x = []  # the label locations
    # label_width = width * len(corr_change_variables)
    # for i in range(len(labels)):
        # x.append(label_width * i)
    # x = np.array(x)


    ax.axhline(linewidth=1, color='k', linestyle='dashed')
    for i, variable in enumerate(corr_change_variables):
        variable_corrs = []
        for index in years:
            variable_corrs.append(correlation_matrices[index]['NOX'][variable])

        ax.plot(years, variable_corrs, '--.', label=variable)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Correlation with NOx')
    ax.set_title('Correlation with NOx of variables by year')
    # ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.legend(title='Variables', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

    fig.tight_layout()
    plt.ylim(-1, 1)
    plt.show()
