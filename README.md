# Box Office Analyser
Developed and trained based on a [Comprehensive Film Statistics Dataset](https://www.kaggle.com/datasets/alessandrolobello/the-ultimate-film-statistics-dataset-for-ml/data), this **Worldwide Gross Revenue ($) Predictive Model 
for Movies** served as my final submission for CAI2C08 Machine Learning. With an approximately **67% Adjusted-R² Value**, my extensive analysis and training catered quite well towards the complexity, depth, and imbalance of this 
dataset. Hosted via Streamlit, my application of this model can be accessed [here](https://sofiaamihan-box-office-analyser-application-rge5mk.streamlit.app/).

Inclusive of my well-researched [Report Analysis](https://github.com/sofiaamihan/box-office-analyser/blob/main/data/dataset-analysis.pdf) of the dataset, I gained valuable insights into the film industry by approaching it from a more analytical and data-driven perspective through the recognition of inherent challenges and unpredictability of box office performances.

![Application Image](https://github.com/sofiaamihan/box-office-analyser/blob/main/data/application.png)

## Features
The Box Office Analyser Repository includes several key features designed to provide insights into movie performance and revenue predictions:
- **Predictive Model**: Utilises my well-trained advanced machine learning algorithms to predict worldwide gross revenue based on various input features.
- **Data Visualisation**: Interactive charts and graphs to visualise relationships between impactful features.
- **Feature Importance Analysis**: Identifies which features most significantly impact revenue predictions, helping users understand the driving factors behind box office success.
- **User-Friendly Interface**: A Streamlit-based application that enables users to input movie data and receive instant predictions and insights.
- **Exploratory Data Analysis (EDA)**: Provides visualisations and statistics to explore the dataset, helping users identify trends and patterns in movie performance.

## Comparison with Deployment Extract 
| Actual Gross Revenue ($)     | Predicted Gross Revenue ($)     | Percentage Difference (%)     |
|------------------------------|---------------------------------|-------------------------------|
| 17, 475, 475                 | 22, 152, 180.67                 | 26.8                          |
| 53, 191, 101                 | 58, 329, 482.08                 | 9.6                           |

## Packages
- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical operations and handling arrays.
- `seaborn`: For data visualisation and statistical graphics.
- `matplotlib`: For creating static, animated, and interactive visualisations.
- `scikit-learn`: For implementing machine learning algorithms and model evaluation.
- `optuna`: For hyperparameter optimisation.
- `joblib`: For saving and loading models.
- `catboost`, `xgboost`, `lightgbm`: For gradient boosting algorithms tailored for regression tasks, all inclusive of my tree-based algorithm choices.

## Setup Instructions
```
pip install -r requirements.txt
```
## Running Locally
```
streamlit run application.py
```
