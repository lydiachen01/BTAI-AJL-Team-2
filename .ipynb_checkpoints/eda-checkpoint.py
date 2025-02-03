import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
train_df = pd.read_csv("train.csv")

# Display basic info
print(train_df.info())
print(train_df.describe())

# Check distribution of skin tones
plt.figure(figsize=(8,5))
sns.countplot(x='fitzpatrick_scale', data=train_df)
plt.title("Distribution of Fitzpatrick Skin Tones")
plt.show()

# Check distribution of labels
plt.figure(figsize=(12,6))
sns.countplot(y='label', data=train_df, order=train_df['label'].value_counts().index)
plt.title("Distribution of Skin Conditions")
plt.show()