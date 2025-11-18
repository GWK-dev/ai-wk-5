# Create a new Jupyter notebook with these cells:

# Cell 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cell 2: Generate sample urban data
np.random.seed(42)
n_cities = 50
urban_data = pd.DataFrame({
    'city': [f'City_{i}' for i in range(n_cities)],
    'population_density': np.random.normal(2000, 800, n_cities),
    'public_transport_usage': np.random.normal(35, 15, n_cities),
    'green_spaces_percent': np.random.normal(25, 10, n_cities),
    'renewable_energy_usage': np.random.normal(20, 12, n_cities)
})

# Cell 3: Calculate sustainability score
urban_data['sustainability_score'] = (
    urban_data['public_transport_usage'] * 0.3 +
    urban_data['green_spaces_percent'] * 0.3 +
    urban_data['renewable_energy_usage'] * 0.4
)

# Cell 4: Display data
urban_data.head()

# Cell 5: Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Sustainability distribution
axes[0,0].hist(urban_data['sustainability_score'], bins=15, alpha=0.7, color='green')
axes[0,0].set_title('Sustainability Score Distribution')

# Correlation heatmap
corr_matrix = urban_data.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr_matrix, annot=True, ax=axes[0,1])
axes[0,1].set_title('Feature Correlation Matrix')

# Public transport vs sustainability
axes[1,0].scatter(urban_data['public_transport_usage'], urban_data['sustainability_score'])
axes[1,0].set_xlabel('Public Transport Usage (%)')
axes[1,0].set_ylabel('Sustainability Score')

# Renewable energy vs sustainability
axes[1,1].scatter(urban_data['renewable_energy_usage'], urban_data['sustainability_score'])
axes[1,1].set_xlabel('Renewable Energy Usage (%)')
axes[1,1].set_ylabel('Sustainability Score')

plt.tight_layout()
plt.show()