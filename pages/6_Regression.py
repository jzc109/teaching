import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import pandas as pd

def generate_data(age_range, num_children, noise_scale):
    np.random.seed(1)
    age = np.random.randint(age_range[0], age_range[1]+1, num_children)
    cognitive_scores = 10 + 2*age + np.random.normal(0, noise_scale, num_children)
    return age, cognitive_scores

def combine_data(age, cognitive_scores):
    combined_age = np.concatenate(age)
    combined_cognitive_scores = np.concatenate(cognitive_scores)
    return combined_age, combined_cognitive_scores

def format_p_value(p_value):
    if p_value < 0.001:
        return "<0.001"
    else:
        return f"{p_value:.2f}"

def plot_scatter(age, cognitive_scores):
    fig, ax = plt.subplots()
    colors = ['red', 'green', 'blue', 'orange']
    r_values = []
    p_values = []
    for i in range(len(age)):
        ax.scatter(age[i], cognitive_scores[i], alpha=0.5, color=colors[i], label=f'Age Group {i+1}')
        slope, intercept, r_value, p_value, _ = linregress(age[i], cognitive_scores[i])
        r_values.append(r_value)
        p_values.append(p_value)
        x = np.linspace(min(age[i]), max(age[i]), 100)
        y = slope * x + intercept
        ax.plot(x, y, color=colors[i], linestyle='--')

    ax.set_xlabel('Age')
    ax.set_ylabel('Cognitive Function')
    ax.set_title('Scatter Plot')

    combined_age, combined_cognitive_scores = combine_data(age, cognitive_scores)
    m, b, r, p, _ = linregress(combined_age, combined_cognitive_scores)
    p_formatted = format_p_value(p)
    ax.plot(combined_age, m * combined_age + b, color='black', label=f'Total (r={r:.2f}, p={p_formatted})')

    handles, labels = ax.get_legend_handles_labels()
    for i in range(len(r_values)):
        p_formatted = format_p_value(p_values[i])
        labels[i] += f' (r={r_values[i]:.2f}, p={p_formatted})'
    ax.legend(handles, labels)

    return fig, r


def app():
    st.title('相关与回归的注意事项')
    st.write('亚组无关，但整体可能相关')

    num_children_per_group = 100

    age_range_group1 = (6, 8)
    age_range_group2 = (9, 11)
    age_range_group3 = (12, 14)
    age_range_group4 = (15, 17)

    age_group1, cognitive_scores_group1 = generate_data(age_range_group1, num_children_per_group, age_noise_scale)
    age_group2, cognitive_scores_group2 = generate_data(age_range_group2, num_children_per_group, age_noise_scale)
    age_group3, cognitive_scores_group3 = generate_data(age_range_group3, num_children_per_group, age_noise_scale)
    age_group4, cognitive_scores_group4 = generate_data(age_range_group4, num_children_per_group, age_noise_scale)

    # Modify the data distributions slightly
    cognitive_scores_group1 += np.random.normal(0, 1, num_children_per_group)
    cognitive_scores_group2 += np.random.normal(0, 2, num_children_per_group)
    cognitive_scores_group3 += np.random.normal(0, 3, num_children_per_group)
    cognitive_scores_group4 += np.random.normal(0, 4, num_children_per_group)

    age = [age_group1, age_group2, age_group3, age_group4]
    cognitive_scores = [cognitive_scores_group1, cognitive_scores_group2,
                        cognitive_scores_group3, cognitive_scores_group4]

    fig, r = plot_scatter(age, cognitive_scores)
    fig.set_size_inches(10, 8)
    st.pyplot(fig)


if __name__ == '__main__':
    age_noise_scale = 8
    app()


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
