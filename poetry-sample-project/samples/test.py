# %% [markdown]


# %%
import tensorflow as tf


# %%
import numpy as np
import matplotlib.pyplot as plt


# %%
x = np.arange(0, 6, 0.1) # 0 から 6 まで 0.1 刻みで生成


# %%
y1 = np.sin(x)
y2 = np.cos(x)

# グラフの描画
plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle = "--", label="cos") # 破線で描画 plt.xlabel("x") # x 軸のラベル
plt.ylabel("y") # y 軸のラベル
plt.title('sin & cos') # タイトル
plt.legend()
plt.show()

# %%
