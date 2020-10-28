import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from itertools import combinations 

def z_normalize_and_seperate_target(df, variable_columns, target_columns):
    # Scale the data on unit scale (mean = 0, variance = 1)
    scaler = StandardScaler()
    # Fit on training set only.
    scaler.fit(df[variable_columns])
    # Apply transform to both the training set and the test set.
    df_scaled = pd.DataFrame(scaler.transform(df[variable_columns]), columns = variable_columns)
    return (df_scaled, df[target_columns])

def z_normalize(train_data, val_data, test_data, feature_names):
    # Scale the data on unit scale (mean = 0, variance = 1)
    scaler = StandardScaler()
    # Fit on training set only.
    scaler.fit(train_data)
    # Apply transform to both the training set and the test set.
    train_data_norm = pd.DataFrame(scaler.transform(train_data), columns = feature_names)
    val_data_norm = pd.DataFrame(scaler.transform(val_data), columns = feature_names)
    test_scaled = pd.DataFrame(scaler.transform(test_data), columns = feature_names)
    return (train_data_norm, val_data_norm, test_scaled)

def compute_performance(pred, observed):
    correlation_df = pd.DataFrame({'NOXa': pred, 'NOXb': observed}, columns=['NOXa', 'NOXb'])
    NOX_correlation = correlation_df.corr(method='spearman').iloc[1][0]
    NOX_mae = mean_absolute_error(observed, pred)
    NOX_r2 = correlation_df.corr(method='pearson').iloc[1][0] ** 2

    return (NOX_correlation, NOX_mae, NOX_r2)

def print_performance(SC, MAE, R2):
    print("NOX Spearman Correlation: " + str(SC))
    print("NOX Mean absolute error: " + str(MAE))
    print("NOX R^2: " + str(R2))

def find_combinations(pairs, k):
    combs = []
    for i in range(0, k):
        comb = list(combinations(pairs, i))
        combs.extend(comb)
    return combs

def frame_operator(A, B):
    return A * B

def apply_engineered_features(dataset, comb):
    for pair in comb:
        featureA = pair[0]
        featureB = pair[1]
        dataset[featureA+featureB] = frame_operator(dataset[featureA], dataset[featureB])
    return dataset

def find_features(baseline, feature_names, train_data, train_out, test_data, test_out):
    print("Finding features...")
    train_data['TEST'] = 0
    test_data['TEST'] = 0
    
    ### Find all pairs of features that when multiplied increase performance
    pairs = []
    for i in range(0, len(feature_names)-1):
        for j in range(i+1, len(feature_names)):
            featureA = feature_names[i]
            featureB = feature_names[j]

            if (featureA == featureB):
                continue
            
            train_data['TEST'] = frame_operator(train_data[featureA], train_data[featureB])
            test_data['TEST'] = frame_operator(test_data[featureA], test_data[featureB])
            
            ### Predict NOX values for the validation data
            # Regress on the training data
            regr = linear_model.LinearRegression()
            regr.fit(train_data, train_out)

            # Predict on test data
            val_pred = regr.predict(test_data)
            val_obs = test_out
            
            (SC, MAE, R2) = compute_performance(featureA+" "+featureB, val_pred, val_obs)
            if (SC > baseline[0] and MAE < baseline[1] and R2 > baseline[2]):
                pairs.append((featureA, featureB))

    # Remove TEST column from both datasets
    del train_data['TEST']
    del test_data['TEST']

    # Compute all combinations of length k of pairs of features
    combs = find_combinations(pairs, 5)
    print(len(pairs))
    print(len(combs))

    ### Regress on all combinations and find best result
    final_MAE = 8.0
    final_data = None
    bestComb = []
    bestDf = None
    for comb in combs:
        final_train_df = train_data.copy()
        final_test_df = test_data.copy()
        for pair in comb:
            final_train_df[pair[0]+pair[1]] = frame_operator(final_train_df[pair[0]], final_train_df[pair[1]])
            final_test_df[pair[0]+pair[1]] = frame_operator(final_test_df[pair[0]], final_test_df[pair[1]])

        ### Predict NOX values for the validation data
        # Regress on the training data
        regr = linear_model.LinearRegression()
        regr.fit(final_train_df, train_out)

        # Predict on test data
        test_pred = regr.predict(final_test_df)
        test_obs = test_out
        
        # Final performance
        (SC, MAE, R2) = compute_performance(featureA+" "+featureB, test_pred, test_obs)
        if (MAE < final_MAE):
            final_MAE = MAE
            final_data = (SC, R2)
            bestComb = comb
            bestDf = final_train_df.copy()
            print(comb)

    print(final_MAE)
    print(final_data)
    print(bestComb)
    print(bestDf)
    return bestComb
