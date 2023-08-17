import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def simulate_sampling_error(true_value, sample_size, num_iterations, std):
    errors = []
    samples = []

    for i in range(num_iterations):
        sample = np.random.normal(loc=true_value, scale=std, size=sample_size)
        sample_mean = np.mean(sample)
        error = sample_mean - true_value

        samples.append(sample_mean)
        errors.append(error)

    return errors, samples

def plot_sampling_error(errors, samples):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].hist(errors, bins=30, edgecolor='black')
    ax[0].set_xlabel('Sampling Error')
    ax[0].set_ylabel('Frequency')
    ax[0].set_title('Sampling Error Distribution')

    ax[1].hist(samples, bins=30, edgecolor='black')
    ax[1].set_xlabel('Sample Value')
    ax[1].set_ylabel('Frequency')
    ax[1].set_title('Sample Value Distribution')

    st.pyplot(fig)

    # Create a dataframe to display the sample means
    data = {'Sample Number': range(1, len(samples)+1),
            'Sample Mean': samples}
    df = pd.DataFrame(data)

    # Display the table
    st.table(df)

    # Calculate and display the mean and standard deviation of the sample means
    mean_of_means = np.mean(samples)
    std_of_means = np.std(samples)
    st.write(f"Mean of Sample Means: {mean_of_means:.2f}")
    st.write(f"Standard Deviation of Sample Means: {std_of_means:.2f}")

def main():
    st.title('Sampling Error Distribution')

    with st.sidebar:
        true_value = st.slider('True Mean', min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        std = st.slider('Standard Deviation', min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        sample_size = st.slider('Sample Size', min_value=10, max_value=1000, value=100, step=10)
        num_iterations = st.slider('Number of Sampling', min_value=10, max_value=1000, value=100, step=10)

    errors, samples = simulate_sampling_error(true_value, sample_size, num_iterations, std)
    plot_sampling_error(errors, samples)

if __name__ == '__main__':
    main()
