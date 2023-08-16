import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t


def plot_distributions(df):
    x = np.linspace(-5, 5, 500)
    y1 = norm.pdf(x)
    y2 = t.pdf(x, df)

    fig, ax = plt.subplots()
    ax.plot(x, y1, label='Standard Normal')
    ax.plot(x, y2, label=f't-distribution (df={df})')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Density')
    ax.set_title('Standard Normal Distribution vs t-distribution')
    st.pyplot(fig)


def main():
    st.title('Standard Normal Distribution vs t-distribution')

    df = st.sidebar.slider('Degrees of Freedom (df)', 1, 300, 1)

    plot_distributions(df)


if __name__ == '__main__':
    main()
