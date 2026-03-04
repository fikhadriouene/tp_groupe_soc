from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from df_utils import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, create_pipeline


def main():
    df = pd.read_csv("data/UNSW_NB15_training-set.csv")
    df_cluster = df[CATEGORICAL_COLUMNS + NUMERIC_COLUMNS]
    clustering_method = KMeans(n_clusters=5, random_state=42, n_init=10)

    pipe = create_pipeline(clustering_method)
    
    y_pred = pipe.fit_predict(df_cluster)

    print(y_pred)

    print(f"Inertie (variance intra-cluster): {clustering_method.inertia_:.2f}")
    print(f"Nombre d'itérations: {clustering_method.n_iter_}")
    print(f"Centres (centroïdes):")
    print(clustering_method.cluster_centers_)


if __name__ == "__main__":
    main()