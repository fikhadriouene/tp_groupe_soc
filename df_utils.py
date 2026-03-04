import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC_COLUMNS = ["dur", "sbytes", "dbytes", "sttl", "dttl", "spkts", "dpkts", "rate"]
CATEGORICAL_COLUMNS = ["proto"]

df = pd.read_csv("data/UNSW_NB15_training-set.csv")
df_cluster = df[CATEGORICAL_COLUMNS + NUMERIC_COLUMNS]

def print_df(df: pd.DataFrame):
    print("Number of rows :",df.shape[0])
    print("Number of columns : ", df.shape[1])
    print("Description :")
    print("Types : ", df.info())
    print("Stats :", df.describe())
    print(df.head(2))

def clean_df(df: pd.DataFrame):
    print("Na values : ", df.isna().sum())
    print("Null values : ", df.isnull().sum())
    df[NUMERIC_COLUMNS]


def detect_outliers_iqr(df, columns):
    outlier_mask = pd.Series(False, index=df.index)

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        mask = (df[col] < lower) | (df[col] > upper)
        outlier_mask = outlier_mask | mask

    return df[outlier_mask]

outliers = detect_outliers_iqr(df, NUMERIC_COLUMNS)
print(outliers)

def remove_outliers(df: pd.DataFrame):
    pass

def create_pipeline(clustering_method)-> Pipeline:
    """ create a pipeline for chosen clustering method """
    preprocessor =  ColumnTransformer(transformers=[
        (
            'num',
            Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]),
            NUMERIC_COLUMNS
        ),
        (
            'cat',
            Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), #dégager avant ?
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]),
            CATEGORICAL_COLUMNS
        )



    ])
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('pca', PCA(n_components=6, random_state=42)),
        ('cluster', clustering_method)
    ])
