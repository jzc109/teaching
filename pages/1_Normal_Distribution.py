import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def plot_normal_distribution(mean, std):
    fig, ax = plt.subplots()
    x = np.linspace(-30, 30, 1000)
    y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-((x - mean) ** 2) / (2 * std ** 2))

    ax.plot(x, y)
    ax.set_xlabel('X')
    ax.set_ylabel('Probability Density')
    ax.set_title('Normal Distribution')
    ax.grid(False)

    return fig, ax

def main():
    st.title('Normal Distribution Simulator')

    # 将滑块放置在侧边栏中
    with st.sidebar:
        mean = st.slider('Mean', min_value=-30.0, max_value=30.0, value=0.0, step=1.0)
        std = st.slider('Standard Deviation', min_value=0.1, max_value=10.0, value=1.0, step=0.1)

    fig, ax = plot_normal_distribution(mean, std)
    st.pyplot(fig)

if __name__ == '__main__':
    main()
