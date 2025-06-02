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

# All the Models I'll be using
from sklearn.tree import DecisionTreeRegressor
import catboost as cb
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

# -------------------- 1. Get the Data --------------------
# Import as variable
df = pd.read_csv('./dataset/movie_statistic_dataset.csv')
# Extract two random rows for deployment testing
extract = df.sample(n=2, random_state=1)
print(extract) # TODO - maybe create a standalone csv file with these values idk
for i in extract.index:
    df.drop(index=i, inplace=True)
# Reset the index of the dataset
df = df.reset_index()
# Modify Feature Headers
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

# -------------------- 2. Visualise the Data --------------------
# ----- Data Cleaning -----
# Removing Unnecessary Columns
del df['movie_title']
# Removing Columns with Unclean values ?? '-'
del df['director_name']
del df['director_professions']
del df['director_birth']
del df['director_death']
# Splitting production_date into 3 separate columns
df['production_date'] = pd.to_datetime(df['production_date'])
# df['day'] = df['production_date'].dt.day
# df['month'] = df['production_date'].dt.month
df['year'] = df['production_date'].dt.year
del df ['production_date']
# Removing Redundant Columns
del df['votes']
del df['rating']
# Prevent Potential Data Leakage
del df['domestic_gross']
# ----- Data Pre-Processing -----
# Binning Runtime Minutes
bin_edges = [90, 150, 210, float('inf')]
bin_labels = ['Short', 'Medium', 'Long']  
df['runtime_category'] = pd.cut(df['runtime_minutes'], bins=bin_edges, labels=bin_labels)