import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna
from time import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
import joblib

from sklearn.tree import DecisionTreeRegressor
import catboost as cb
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

# -------------------- 1. Get the Data --------------------

df = pd.read_csv('./dataset/movie_statistic_dataset.csv')

deployment_extract = df.sample(n=2, random_state=1)
deployment_extract.to_csv('deployment_extract.csv', index=False)
for i in deployment_extract.index:
    df.drop(index=i, inplace=True)

df = df.reset_index()
del df['index']
renamed_columns = {
    'director_birthYear' : 'director_birth', 
    'director_deathYear': 'director_death', 
    'movie_averageRating': 'rating', 
    'movie_numerOfVotes': 'votes', 
    'approval_Index': 'approval_index', 
    'Production budget $': 'budget', 
    'Domestic gross $': 'domestic_gross', 
    'Worldwide gross $': 'worldwide_gross' 
}
df.rename(renamed_columns, axis='columns', inplace=True)
print(df['production_date'])

# -------------------- 2. Visualise the Data --------------------

news_row = df[df['genres'].str.contains('News', na=False)]
df.drop(index=news_row.index, inplace=True)
df = df.reset_index()
del df['index'] 

del df['movie_title']
del df['director_name']
del df['director_professions']
del df['director_birth']
del df['director_death']

df['production_date'] = pd.to_datetime(df['production_date'])
df['year'] = df['production_date'].dt.year
del df ['production_date']

del df['votes']
del df['rating']
del df['domestic_gross']

bin_edges = [0, 90, 150, 210, float('inf')]
bin_labels = ['Short', 'Medium', 'Long', 'VeryLong']  
df['runtime_category'] = pd.cut(df['runtime_minutes'], bins=bin_edges, labels=bin_labels)
df_cat = pd.get_dummies(df, columns=['runtime_category'], drop_first=True)
df_cat = df_cat.astype({ 'runtime_category_Medium': 'int','runtime_category_Long': 'int', 'runtime_category_VeryLong': 'int'})
del df_cat['runtime_minutes']
df = df_cat

# -------------------- 3. Shuffle and Split the Data --------------------

y = df['worldwide_gross'].values
del df['worldwide_gross']
X = df.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.Series(y_train)
y_test = pd.Series(y_test)

renamed_columns = {
    0:'genres', 
    1:'approval_index',
    2:'budget',
    3:'year', 
    4:'runtime_category_Medium', 
    5:'runtime_category_Long', 
    6:'runtime_category_VeryLong'
}
X_train.rename(renamed_columns, axis='columns', inplace=True)
X_test.rename(renamed_columns, axis='columns', inplace=True)

robust_scaler = RobustScaler()
X_train[['budget_scaled']] = robust_scaler.fit_transform(X_train[['budget']])
X_train.drop(columns=['budget'], inplace=True)
s_scaler = StandardScaler()
X_train[['year_scaled']] = s_scaler.fit_transform(X_train[['year']])
X_train.drop(columns=['year'], inplace=True)
new = X_train['genres'].str.get_dummies(sep=',')
del X_train['genres']
pca = PCA(n_components=5)
pca.fit(new)
new_data = pd.DataFrame(pca.transform(new), columns=['pc1', 'pc2', 'pc3', 'pc4', 'pc5'])
X_train = pd.DataFrame(X_train).join(new_data)

X_test[['budget_scaled']] = robust_scaler.transform(X_test[['budget']])  
X_test.drop(columns=['budget'], inplace=True)
X_test[['year_scaled']] = s_scaler.transform(X_test[['year']])  
X_test.drop(columns=['year'], inplace=True)
genre_test = X_test['genres'].str.get_dummies(sep=',')
genre_pca_test = pd.DataFrame(pca.transform(genre_test), columns=['pc1', 'pc2', 'pc3', 'pc4', 'pc5'])
X_test.drop(columns=['genres'], inplace=True)
X_test = X_test.join(genre_pca_test)

X_train['approval_index'] = pd.to_numeric(X_train['approval_index'], errors='coerce')  
X_train['budget_scaled'] = pd.to_numeric(X_train['budget_scaled'], errors='coerce')
X_train['year_scaled'] = pd.to_numeric(X_train['year_scaled'], errors='coerce')
X_train['runtime_category_Medium'] = pd.to_numeric(X_train['runtime_category_Medium'], errors='coerce')
X_train['runtime_category_Long'] = pd.to_numeric(X_train['runtime_category_Long'], errors='coerce')
X_train['runtime_category_VeryLong'] = pd.to_numeric(X_train['runtime_category_VeryLong'], errors='coerce')

X_test['approval_index'] = pd.to_numeric(X_test['approval_index'], errors='coerce') 
X_test['budget_scaled'] = pd.to_numeric(X_test['budget_scaled'], errors='coerce')
X_test['year_scaled'] = pd.to_numeric(X_test['year_scaled'], errors='coerce')
X_test['runtime_category_Medium'] = pd.to_numeric(X_test['runtime_category_Medium'], errors='coerce')
X_test['runtime_category_Long'] = pd.to_numeric(X_test['runtime_category_Long'], errors='coerce')
X_test['runtime_category_VeryLong'] = pd.to_numeric(X_test['runtime_category_VeryLong'], errors='coerce')

# -------------------- 5. Hyperparameters --------------------
start = time()
speed = {}

def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 1000, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'depth': trial.suggest_int('depth', 6, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 3.0),
        'border_count': trial.suggest_int('border_count', 200, 255),
        'subsample': trial.suggest_float('subsample', 0.7, 0.9),
        'random_strength': trial.suggest_int('random_strength', 1.0, 3.0),
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_maes = [] 
    for train_idx, val_idx in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model = cb.CatBoostRegressor(random_state=42, silent=True, **params)
        model.fit(X_train_fold, y_train_fold, eval_set=(X_val_fold, y_val_fold), verbose=0)
        mae = mean_absolute_error(y_val_fold, model.predict(X_val_fold))
        fold_maes.append(mae)
    return np.mean(fold_maes)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
best_params = study.best_params
print("Best parameters:", best_params)

speed['CatBoosting'] = np.round(time()-start, 3)
print(f"Run time: {speed['CatBoosting']}s")

# -------------------- 6. Train the Model --------------------
model = cb.CatBoostRegressor(random_state=42, silent=True, **best_params)
model.fit(X_train, y_train)

# -------------------- 7. Evaluate the Model --------------------
mae_train = mean_absolute_error(y_train, model.predict(X_train))
mae_test = mean_absolute_error(y_test, model.predict(X_test))

r_squared = r2_score(y_test, model.predict(X_test))
mse = mean_squared_error(y_test, model.predict(X_test))
rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

def adjusted_r_squared(y_true, y_pred, num_features):
    n = len(y_true)
    return 1 - ((1 - r_squared) * (n - 1) / (n - num_features - 1))

num_features = len(best_params)
adj_r_squared = adjusted_r_squared(y_test, model.predict(X_test), num_features)
train_adj_r_squared = adjusted_r_squared(y_train, model.predict(X_train), num_features)

print("MAE on training data:", mae_train)
print("MAE on testing data:", mae_test)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r_squared)
print("Adjusted R-squared:", adj_r_squared)
print("Adjusted R-squared FOR TRAIN:", train_adj_r_squared)

speed['CatBoost'] = np.round(time() - start, 3)
print(f"Run time: {speed['CatBoost']}s")

# -------------------- 8. Deploy the Model --------------------
joblib.dump(robust_scaler, 'robust_scaler.pkl')
joblib.dump(s_scaler, 'standard_scaler.pkl')
joblib.dump(pca, 'genre_pca.pkl')
joblib.dump(model, 'trained_box_office_analyser.pkl')