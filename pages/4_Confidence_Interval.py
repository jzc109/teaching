import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def plot_confidence_intervals(mu, sigma, n, num_samples, confidence_level):
    # 计算置信区间的 Z 值
    if confidence_level == '80%':
        z_score = 1.28
    elif confidence_level == '85%':
        z_score = 1.44
    elif confidence_level == '90%':
        z_score = 1.645
    elif confidence_level == '95%':
        z_score = 1.96
    elif confidence_level == '99%':
        z_score = 2.576
    else:
        raise ValueError('Invalid confidence level')

    # 生成正态分布随机数
    samples = np.random.normal(mu, sigma, size=(num_samples, n))
    # 计算样本均值和标准误差
    means = np.mean(samples, axis=1)
    std_errs = np.std(samples, axis=1, ddof=1) / np.sqrt(n)
    # 计算置信区间
    cis = np.column_stack((means - z_score * std_errs, means + z_score * std_errs))
    # 绘制置信区间森林图
    fig, ax = plt.subplots()
    for i in range(num_samples):
        if mu < cis[i, 0] or mu > cis[i, 1]:
            color = 'r'
        else:
            color = 'k'
        ax.plot(cis[i], [i, i], color=color, linewidth=0.5)
        ax.plot(means[i], i, 'o', color='k', markersize=2)
    ax.axvline(mu, color='b', linestyle='--', linewidth=1)
    ax.set_yticks([0])
    ax.set_yticklabels([''])
    ax.set_xticks([mu - 3*sigma/ np.sqrt(n), mu - 2*sigma/ np.sqrt(n), mu - sigma/ np.sqrt(n), mu, mu + sigma/ np.sqrt(n), mu + 2*sigma/ np.sqrt(n), mu + 3*sigma/ np.sqrt(n)])
    ax.set_xlabel('Value')
    ax.set_title('Confidence Intervals')
    fig.set_size_inches(10, 8)
    st.pyplot(fig)



# 设置页面标题和侧边栏
st.set_page_config(page_title='Confidence Intervals', page_icon=':bar_chart:', layout='wide')
st.sidebar.title('Parameters')

# 添加滑动条和下拉框控件
mu = st.sidebar.slider('Mean', -10.0, 10.0, 0.0, 0.1)
sigma = st.sidebar.slider('Standard Deviation', 0.1, 5.0, 1.0, 0.1)
n = st.sidebar.slider('Sample Size', 10, 1000, 100, 10)
num_samples = st.sidebar.slider('Number of Samples', 10, 1000, 100, 10)
confidence_level = st.sidebar.selectbox('Confidence Level', ['80%', '85%', '90%', '95%', '99%'])

# 绘制置信区间森林图
plot_confidence_intervals(mu, sigma, n, num_samples, confidence_level)
