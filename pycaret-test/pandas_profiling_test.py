#%%
import pandas as pd
import pandas_profiling as pdp
from sklearn import datasets

# %%
data = datasets.load_boston()
# %%
df = pd.DataFrame(data.data, columns=data.feature_names)
# %%
profile = pdp.ProfileReport(df)
# %%
profile.to_file(output_file="pandas_profiling_test.html")
