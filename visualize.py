import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load crowd detection results
df_crowds = pd.read_csv("crowd_detections.csv")

# Set up the figure
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

# Smooth trend line
sns.lineplot(data=df_crowds, x="Frame", y="Person_Count", color="#007bff", lw=2, label="Crowd Size")

# Highlight peaks
max_persons = df_crowds["Person_Count"].max()
peaks = df_crowds[df_crowds["Person_Count"] == max_persons]
plt.scatter(peaks["Frame"], peaks["Person_Count"], color='red', edgecolors='black', s=100, label="Peak Crowds")

# Labels & Title
plt.xlabel("Frame Number", fontsize=12)
plt.ylabel("Number of People", fontsize=12)
plt.title("📊 Crowd Size Over Time", fontsize=14, fontweight='bold')

# Grid & Legend
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

plt.show()
