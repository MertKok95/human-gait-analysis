import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Loading the dataset
dataframe_gait = pd.read_csv("../data/gait_parameters.csv")

# 2. Visualization style (Clean, white background)
sns.set_theme(style="whitegrid")

# 3. Creating a figure with 1 row and 3 columns.
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Our color palette (Blue for normal walking, red for fast walking to create contrast)
palette = ["#3498db", "#e74c3c"]

# --- Figure 1: Velocity Comparison ---
sns.boxplot(data=dataframe_gait[['Velocity_UGS', 'Velocity_FGS']], palette=palette, ax=axes[0])
axes[0].set_title('Velocity Comparison', fontsize=14, fontweight='bold')
axes[0].set_xticklabels(['Usual Gait (UGS)', 'Fast Gait (FGS)'])
axes[0].set_ylabel('Speed (m/s)')

# --- Figure 2: Cadence (Step/Minute) Comparison ---
sns.boxplot(data=dataframe_gait[['Cadence_UGS', 'Cadence_FGS']], palette=palette, ax=axes[1])
axes[1].set_title('Cadence Comparison', fontsize=14, fontweight='bold')
axes[1].set_xticklabels(['Usual Gait (UGS)', 'Fast Gait (FGS)'])
axes[1].set_ylabel('Steps per minute')

# --- Figure 3: Stride Length Comparison ---
sns.boxplot(data=dataframe_gait[['Stride_UGS', 'Stride_FGS']], palette=palette, ax=axes[2])
axes[2].set_title('Stride Length Comparison', fontsize=14, fontweight='bold')
axes[2].set_xticklabels(['Usual Gait (UGS)', 'Fast Gait (FGS)'])
axes[2].set_ylabel('Length (cm)')

# 4. Editing and Saving
plt.tight_layout(pad=6.0, h_pad=0.5, w_pad=2.0) 
plt.savefig('../data/outputs/figures/diagrams/gait_parameters_comparison.png', dpi=300)
print("saved location: ../data/outputs/figures/diagrams/gait_parameters_comparison.png")
# plt.show()