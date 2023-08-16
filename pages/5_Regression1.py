import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

st.title('相关与回归的注意事项')
st.write('亚组相关，但整体可能不相关')
# Create data for different age groups
np.random.seed(0)
age_groups = [(6, 8), (9, 11), (12, 14), (15, 17)]
subset_data = []
subset_stats = []
for age_group in age_groups:
    np.random.seed(age_group[0])
    age = np.random.randint(age_group[0], age_group[1] + 1, size=100)
    mean = [age_group[0], 300]
    # Modify the covariance matrix to adjust correlation within each age group
    cov = [[1, 0.55], [0.55, 1]]
    data = np.random.multivariate_normal(mean, cov, size=100)
    subset = pd.DataFrame({
        'Age': data[:, 0],
        'Lung Capacity': data[:, 1]
    })
    subset_data.append(subset)

    # Calculate correlation coefficient and regression line for each subset
    slope, intercept, r_value, p_value, _ = linregress(subset['Age'], subset['Lung Capacity'])
    if p_value < 0.001:
        p_value_text = "p < 0.001"
    else:
        p_value_text = f"p = {p_value:.3f}"
    subset_stats.append({
        'Age Group': f"{age_group[0]}-{age_group[1]} Years",
        'Slope': slope,
        'Intercept': intercept,
        'R': r_value,
        'P-value': p_value_text
    })

# Combine the data subsets
data = pd.concat(subset_data)

# Plot scatter plot and regression lines
fig, ax = plt.subplots(figsize=(10, 6))
for subset, stats in zip(subset_data, subset_stats):
    label = f"{stats['Age Group']} (r: {stats['R']:.2f}, {stats['P-value']})"
    ax.scatter(subset['Age'], subset['Lung Capacity'], label=label)
    x = np.linspace(subset['Age'].min(), subset['Age'].max(), 100)
    y = stats['Slope'] * x + stats['Intercept']
    ax.plot(x, y, color='red', linestyle='--')
ax.legend()
ax.set_xlabel('Age')
ax.set_ylabel('Lung Capacity')

# Calculate correlation coefficient and regression line for the entire dataset
slope, intercept, r_value, p_value, _ = linregress(data['Age'], data['Lung Capacity'])
if p_value < 0.001:
    p_value_text = "p < 0.001"
else:
    p_value_text = f"p = {p_value:.3f}"
total_stats = {
    'Slope': slope,
    'Intercept': intercept,
    'R': r_value,
    'P-value': p_value_text
}
total_label = f"Total (r: {total_stats['R']:.2f}, {total_stats['P-value']})"
x = np.linspace(data['Age'].min(), data['Age'].max(), 100)
y = total_stats['Slope'] * x + total_stats['Intercept']
ax.plot(x, y, color='blue', linestyle='--', label=total_label)
ax.legend()

# Adjust title font size
plt.title('Scatter Plot and Regression Lines', fontsize=10)

# Adjust figure size
fig.set_size_inches(10, 8)

# Display the plot and statistics in Streamlit
st.pyplot(fig)
st.write("Subset Stats:")
subset_stats_df = pd.DataFrame(subset_stats)
st.write(subset_stats_df)
st.write("Total Stats:")
total_stats_df = pd.DataFrame(total_stats, index=[0])
st.write(total_stats_df)