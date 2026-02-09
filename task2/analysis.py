import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("unemployment.csv")

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

df["date"] = pd.to_datetime(df["date"], dayfirst=True)

df["estimated_unemployment_rate_(%)"] = pd.to_numeric(
    df["estimated_unemployment_rate_(%)"], errors="coerce"
)

df = df.dropna(subset=["date", "estimated_unemployment_rate_(%)"])

df = df.groupby("date", as_index=False)["estimated_unemployment_rate_(%)"].mean()

df = df.sort_values("date")

plt.figure()
plt.plot(df["date"], df["estimated_unemployment_rate_(%)"])
plt.title("Unemployment Rate Over Time (India)")
plt.xlabel("Year")
plt.ylabel("Unemployment Rate (%)")
plt.show()

df_area = pd.read_csv("unemployment.csv")
df_area.columns = df_area.columns.str.strip().str.lower().str.replace(" ", "_")
df_area["date"] = pd.to_datetime(df_area["date"], dayfirst=True)
df_area["estimated_unemployment_rate_(%)"] = pd.to_numeric(
    df_area["estimated_unemployment_rate_(%)"], errors="coerce"
)

area_trend = df_area.groupby(["date", "area"])["estimated_unemployment_rate_(%)"].mean().unstack()

area_trend.plot()
plt.title("Urban vs Rural Unemployment")
plt.show()

