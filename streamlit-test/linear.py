import streamlit as st
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.datasets import load_boston #ボストン市の住宅価格データ
import seaborn as sns

"""
# ボストン市の住宅価格データ
"""
dataset = load_boston()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)#説明変数
df['MEDV_PRICE'] = dataset.target #目的変数を追加
df

#pg = sns.pairplot(df)
#st.pyplot(pg)
from yellowbrick.datasets import load_credit
from yellowbrick.features import Rank1D

# Load the credit dataset
X, y = load_credit()

#visualizer = Rank2D(features=dataset.features, algorithm='pearson')
#visualizer.fit(dataset.X, dataset.y)
#visualizer.transform(X);