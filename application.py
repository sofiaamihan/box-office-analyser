import streamlit as st
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
import joblib

# -------------------- Header --------------------
st.image('./data/little_women.png')
st.markdown(
    f"""
    <div style='display: flex; justify-content: center;'>
        <h1>Box Office Analyser</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------- Sidebar --------------------
st.sidebar.header('Movie Details')

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
    'Romance',
    'Sci-Fi',
    'Sport',
    'Thriller',
    'War',
    'Western',
    '\\N'
]

def user_input_features():
    budget = st.sidebar.slider('Production Budget ($)', 50000, 460000000, step=1)
    year = st.sidebar.slider('Production Year', 1915, 2023, step=1)
    runtime_minutes = st.sidebar.slider('Movie Duration (Mins)', 63, 271, step=1)
    approval_index = st.sidebar.slider('Movie Approval Index', 0.40, 10.00)
    movie_genres = st.sidebar.multiselect(label='Select Movie Genres', options=genres)
    genre = ','.join(movie_genres)

    data = {
        'budget': budget,
        'year': year,
        'runtime_minutes': runtime_minutes,
        'approval_index': approval_index,
        'genres': genre
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

# -------------------- Body --------------------
df = user_input_features()
genre_display = f"{df['genres'].values[0]}".replace(",", ", ")
st.markdown(
    f"""
    <div style='display: flex; justify-content: space-between;'>
        <strong>Budget:</strong> <span>${df['budget'].values[0]:,.2f}</span>
    </div>
    <div style='display: flex; justify-content: space-between;'>
        <strong>Production Year:</strong> <span>{df['year'].values[0]}</span>
    </div>
    <div style='display: flex; justify-content: space-between;'>
        <strong>Movie Duration:</strong> <span>{df['runtime_minutes'].values[0]} Minutes</span>
    </div>
    <div style='display: flex; justify-content: space-between;'>
        <strong>Genres:</strong> <span>{genre_display}</span>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------- Data Processing --------------------
robust_scaler = joblib.load('robust_scaler.pkl')
s_scaler = joblib.load('standard_scaler.pkl')
pca = joblib.load('genre_pca.pkl')
model = joblib.load('trained_box_office_analyser.pkl')

df[['budget_scaled']] = robust_scaler.transform(df[['budget']])
df.drop(columns=['budget'], inplace=True)

df[['year_scaled']] = s_scaler.transform(df[['year']])
df.drop(columns=['year'], inplace=True)

bin_edges = [0, 90, 150, 210, float('inf')]
bin_labels = ['Short', 'Medium', 'Long', 'VeryLong']  
df['runtime_category'] = pd.cut(df['runtime_minutes'], bins=bin_edges, labels=bin_labels)
df_cat = pd.get_dummies(df, columns=['runtime_category'], drop_first=True)
df_cat = df_cat.astype({ 'runtime_category_Medium': 'int','runtime_category_Long': 'int', 'runtime_category_VeryLong': 'int'})
del df_cat['runtime_minutes']
df = df_cat

genre_vector = pd.DataFrame([[1 if genre in df['genres'][0].split(',') else 0 for genre in genres]],columns=genres)
del df['genres']
new_data = pd.DataFrame(pca.transform(genre_vector), columns=['pc1', 'pc2', 'pc3', 'pc4', 'pc5'])
df = pd.DataFrame(df).join(new_data)

# -------------------- Output --------------------
prediction = model.predict(df)
revenue = f"{abs(prediction[0]):,.2f}".replace(",", ", ")
st.markdown(
    f"""
    <div style='display: flex; justify-content: center;'>
        <span><h4>Predicted Worldwide Gross Revenue:</h4></span> <span><h4>${revenue}</h4></span>
    </div>
    """,
    unsafe_allow_html=True
)