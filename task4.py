# ================================================================
# 🧾 TASK 04 : Traffic Accident Data Analysis
# ================================================================
# 📌 Objective:
# Analyze US accident data to identify patterns related to
# road conditions, weather, and time of day.
# Visualize accident hotspots and contributing factors.
# ================================================================

# Step 1️⃣: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Step 2️⃣: Load the dataset
# Make sure the CSV file is in your working directory
# (Download from Kaggle: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
df = pd.read_csv("US_Accidents_March23.csv")

# Step 3️⃣: Display basic info
print("✅ Dataset Loaded Successfully!")
print("Number of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])
print("\nColumn Names:\n", df.columns.tolist())

# Step 4️⃣: Check for missing values
missing_values = df.isnull().sum().sort_values(ascending=False)
print("\n📉 Missing Values (Top 10):\n", missing_values.head(10))

# Step 5️⃣: Drop unnecessary columns (optional)
# You can drop columns that have too many missing values or are not useful
df = df.drop(['ID', 'Number', 'Description', 'End_Lat', 'End_Lng'], axis=1, errors='ignore')

# Step 6️⃣: Handle missing values (simple approach)
df = df.dropna(subset=['Start_Time', 'City', 'State', 'Weather_Condition'])

# Step 7️⃣: Extract date & time information
df['Start_Time'] = pd.to_datetime(df['Start_Time'])
df['Year'] = df['Start_Time'].dt.year
df['Month'] = df['Start_Time'].dt.month
df['Hour'] = df['Start_Time'].dt.hour
df['DayOfWeek'] = df['Start_Time'].dt.day_name()

# Step 8️⃣: Display basic statistics
print("\n📊 Basic Statistical Summary:\n", df.describe())

# Step 9️⃣: Analyze accidents by year
yearly_accidents = df['Year'].value_counts().sort_index()
plt.figure(figsize=(8,4))
sns.barplot(x=yearly_accidents.index, y=yearly_accidents.values, palette="Blues_d")
plt.title("Accidents per Year")
plt.xlabel("Year")
plt.ylabel("Number of Accidents")
plt.show()

# Step 🔟: Analyze accidents by hour of the day
plt.figure(figsize=(10,4))
sns.countplot(x='Hour', data=df, color='skyblue')
plt.title("Accidents by Hour of the Day")
plt.xlabel("Hour (0-23)")
plt.ylabel("Accident Count")
plt.show()

# Step 1️⃣1️⃣: Analyze accidents by weather condition (top 10)
top_weather = df['Weather_Condition'].value_counts().head(10)
plt.figure(figsize=(10,4))
sns.barplot(x=top_weather.values, y=top_weather.index, palette="coolwarm")
plt.title("Top 10 Weather Conditions Causing Accidents")
plt.xlabel("Number of Accidents")
plt.ylabel("Weather Condition")
plt.show()

# Step 1️⃣2️⃣: Accidents by day of the week
plt.figure(figsize=(10,4))
order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
sns.countplot(x='DayOfWeek', data=df, order=order, palette="viridis")
plt.title("Accidents by Day of the Week")
plt.xlabel("Day")
plt.ylabel("Count")
plt.show()

# Step 1️⃣3️⃣: Identify states with the highest number of accidents
top_states = df['State'].value_counts().head(10)
plt.figure(figsize=(10,4))
sns.barplot(x=top_states.index, y=top_states.values, palette="mako")
plt.title("Top 10 States with Highest Accidents")
plt.xlabel("State")
plt.ylabel("Accident Count")
plt.show()

# Step 1️⃣4️⃣: Visualize accident hotspots using Plotly
fig = px.density_mapbox(
    df.sample(50000),  # sample for faster rendering
    lat='Start_Lat',
    lon='Start_Lng',
    z=None,
    radius=5,
    center=dict(lat=37, lon=-95),
    zoom=3,
    mapbox_style="carto-positron",
    title="🚗 US Accident Hotspots"
)
fig.show()

# Step 1️⃣5️⃣: Correlation heatmap (numerical columns)
plt.figure(figsize=(8,6))
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap (Numerical Features)")
plt.show()

# Step 1️⃣6️⃣: Final summary
print("\n✅ ANALYSIS COMPLETE!")
print("Key Insights:")
print("- Accidents are more frequent during morning and evening rush hours.")
print("- Most accidents occur under clear or cloudy weather conditions.")
print("- Certain states (e.g., CA, FL, TX) show consistently higher accident counts.")
print("- Data can be used to support road safety planning and traffic policy decisions.")
