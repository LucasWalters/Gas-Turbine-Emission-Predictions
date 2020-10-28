import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

def z_normalize_and_seperate_target(df, variable_columns, target_columns):
    # Scale the data on unit scale (mean = 0, variance = 1)
    scaler = StandardScaler()
    # Fit on training set only.
    scaler.fit(df[variable_columns])
    # Apply transform to both the training set and the test set.
    df_scaled = pd.DataFrame(scaler.transform(df[variable_columns]), columns = variable_columns)
    return (df_scaled, df[target_columns])

def compute_performance(name, pred, observed):
    correlation_df = pd.DataFrame({'NOXa': pred, 'NOXb': observed}, columns=['NOXa', 'NOXb'])
    NOX_correlation = correlation_df.corr(method='spearman').iloc[1][0]
    NOX_mae = mean_absolute_error(observed, pred)
    NOX_r2 = correlation_df.corr(method='pearson').iloc[1][0] ** 2

    return (NOX_correlation, NOX_mae, NOX_r2)

def print_performance(SC, MAE, R2):
    print("NOX Spearman Correlation: " + str(SC))
    print("NOX Mean absolute error: " + str(MAE))
    print("NOX R^2: " + str(R2))

def find_features(feature_names, train_df, train_Y, test_df, test_Y):
    train_df['TEST'] = 0
    test_df['TEST'] = 0
    
    pairs = []
    for i in range(0, len(feature_names)-1):
        for j in range(i+1, len(feature_names)):
            featureA = feature_names[i]
            featureB = feature_names[j]

            if (featureA == featureB):
                continue
            
            #print(featureA + " " + featureB)
            train_df['TEST'] = train_df[featureA] * train_df[featureB]
            test_df['TEST'] = test_df[featureA] * test_df[featureB]
            
            ### Predict NOX values for the validation data
            # Regress on the training data
            regr = linear_model.LinearRegression()
            regr.fit(train_df, train_Y)

            # Predict on test data
            val_pred = regr.predict(test_df)
            val_obs = test_Y
            
            (SC, MAE, R2) = compute_performance(featureA+" "+featureB, val_pred, val_obs)
            if (SC > 0.4951 and MAE < 7.6895 and R2 > 0.2745):
                pairs.append((featureA, featureB))
                print_performance(SC, MAE, R2)
    print(len(pairs))
