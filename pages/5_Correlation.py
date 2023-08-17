import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 设置页面宽度
st.set_page_config(layout="wide")

# 标题
st.title("模拟线性相关散点图")

# 设置线性相关系数的滑块
corr_coef = st.slider("线性相关系数", -1.0, 1.0, step=0.1)

# 生成随机数据
np.random.seed(0)
mean = [0, 0]
cov = [[1, corr_coef], [corr_coef, 1]]
x, y = np.random.multivariate_normal(mean, cov, 200).T

# 计算相关系数
correlation_coef = np.corrcoef(x, y)[0, 1]

# 绘制散点图
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x, y)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title(f"Scatter Plot ")
# ax.set_title(f"Scatter Plot (Correlation Coefficient: {correlation_coef:.2f})")
# 显示散点图
st.pyplot(fig)
