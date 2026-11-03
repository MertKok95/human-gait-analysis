import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Install dataset
# Don't forget to update the file path according to your own folder structure.
dataframe_patients = pd.read_csv("../data/patients_measures.csv")

# 2. Making the gender column more readable for visualization purposes.
# According to the README file: 0 = Female, 1 = Male
dataframe_patients['Sex_Label'] = dataframe_patients['Sex'].map({0: 'Female', 1: 'Male'})

# 3. Determining the visualization style 
sns.set_theme(style="whitegrid")

# 4. Creating a figure (canvas) for 3 graphs side by side.
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- Figure 1: Age Distribution ---
# Kernel Density Estimation (kde)
sns.histplot(data=dataframe_patients, x='Age', kde=True, color='skyblue', ax=axes[0])
axes[0].lines[0].set_color('red')
axes[0].set_title('Age Distribution of Participants', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Age (Years)')
axes[0].set_ylabel('Count')

# --- Figure 2: Gender Distribution ---
sns.countplot(data=dataframe_patients, x='Sex_Label', hue='Sex_Label', palette={'Female': '#ffb6c1', 'Male': '#add8e6'}, ax=axes[1], legend=False)
axes[1].set_title('Gender Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Gender')
axes[1].set_ylabel('Count')

# --- Figure 3: BMI (Body Mass Index) Distribution ---
sns.histplot(data=dataframe_patients, x='BMI', kde=True, color='salmon', ax=axes[2])
axes[2].set_title('BMI Distribution', fontsize=14, fontweight='bold')
axes[2].lines[0].set_color('blue')
axes[2].set_xlabel('BMI (kg/m²)')
axes[2].set_ylabel('Count')

# 5. Editing and saving graphics
plt.tight_layout(pad=4.0, h_pad=0.5, w_pad=0.5) # It prevents the graphs from overlapping.
plt.savefig('../data/outputs/figures/diagrams/patient_demographic_distributions.png', dpi=300) 
print("saved location: ../data/outputs/figures/diagrams/patient_demographic_distributions.png")
# plt.show()