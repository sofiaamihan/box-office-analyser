import streamlit as st
import pandas as pd
import joblib
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

st.write("""
# Box Office Analyser - Worldwide Gross $ Predictor
This application predicts the Worldwide Gross Revenue of your movie!
""")

st.sidebar.header('User Input Parameters')

genres = [
    'Action',
    'Adventure',
    'Animation',
    'Biography',
    'Comedy',
    'Crime',
    'Documentary',
    'Drama',
    'Family',
    'Fantasy',
    'Film-Noir',
    'History',
    'Horror',
    'Music',
    'Musical',
    'Mystery',
    'News',
    'Romance',
    'Sci-Fi',
    'Sport',
    'Thriller',
    'War',
    'Western'
]


def user_input_features():
    budget = st.sidebar.slider('Production Budget ($)', 50000.00, 460000000.00)
    year = st.sidebar.slider('Release Year', 1915, 2023, step=1)
    runtime_minutes = st.sidebar.slider('Movie Duration in Minutes', 63.0, 271.00)
    approval_index = st.sidebar.slider('Movie Approval Index', 0.40, 10.00)
    genre = st.sidebar.selectbox(
        'Select Movie Genre',
        [
            'Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
            'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'History',
            'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance', 'Sci-Fi',
            'Sport', 'Thriller', 'War', 'Western', '\\N',
        ]
    )
    # genre_selection = []
    # for genre in genres:
    #     st.sidebar.checkbox(genre)
    st.sidebar.multiselect(label='Select Genres', options=genres)
    
    data = {
        'budget': budget,
        'year': year,
        'runtime_minutes': runtime_minutes,
        'approval_index': approval_index,
        'genre': genre
    }
    
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

model = joblib.load('trained_box_office_analyser.pkl')

robust_scaler = RobustScaler()
df[['budget_scaled']] = robust_scaler.fit_transform(df[['budget']])
df.drop(columns=['budget'], inplace=True)

s_scaler = StandardScaler()
df[['year_scaled']] = s_scaler.fit_transform(df[['year']])
df.drop(columns=['year'], inplace=True)

bin_edges = [0, 90, 150, 210, float('inf')]
bin_labels = ['Short', 'Medium', 'Long', 'VeryLong']  
df['runtime_category'] = pd.cut(df['runtime_minutes'], bins=bin_edges, labels=bin_labels)
df_cat = pd.get_dummies(df, columns=['runtime_category'], drop_first=True)
df_cat = df_cat.astype({ 'runtime_category_Medium': 'int','runtime_category_Long': 'int', 'runtime_category_VeryLong': 'int'})
del df_cat['runtime_minutes']
df = df_cat

def genre_to_onehot(selected_genre):
    all_genres = [
        'Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
        'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'History',
        'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance', 'Sci-Fi',
        'Sport', 'Thriller', 'War', 'Western', '\\N'
    ]
    
    genre_vector = [0] * len(all_genres)
    
    if selected_genre in all_genres:
        genre_index = all_genres.index(selected_genre)
        genre_vector[genre_index] = 1
    
    return genre_vector


genre_vector = genre_to_onehot(df['genre'][0])
genre_all = pd.DataFrame([genre_vector] * 5, columns=[
    'Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'History',
    'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance', 'Sci-Fi',
    'Sport', 'Thriller', 'War', 'Western', '\\N'
])
pca = PCA(n_components=5)
pca.fit(genre_all)
new_data = pd.DataFrame(pca.transform(genre_all), columns=['pc1', 'pc2', 'pc3', 'pc4', 'pc5'])
user_data = new_data.iloc[0]
final_data = pd.DataFrame({
    'pc1': [user_data['pc1']],
    'pc2': [user_data['pc2']],
    'pc3': [user_data['pc3']],
    'pc4': [user_data['pc4']],
    'pc5': [user_data['pc5']]
})
del df['genre']

df = df.join(final_data)

prediction = model.predict(df)

st.subheader('Predicted Worldwide Gross $')
st.write(prediction[0])














