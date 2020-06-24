# %%
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# In[108]:
iris = load_iris()
X = iris.data
print(X)

# %%
y = iris.target
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# %%
lr = LogisticRegression()
lr.fit(X_train, Y_train)

# %%
# lr.predict_proba(X_test[0, :])
