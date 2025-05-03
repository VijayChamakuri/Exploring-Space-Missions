import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from calendar import month_abbr
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide")
sns.set_theme()

st.title("🚀 Exploring Space Missions")

@st.cache_data
def load_data():
    csv_path = "/workspaces/Exploring-Space-Missions/Space_Corrected.csv"
    df = pd.read_csv(csv_path)
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    # Drop any Unnamed columns
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    # Parse dates & extract year/month/day
    df['Datum'] = pd.to_datetime(df['Datum'], errors='coerce')
    df['year']  = df['Datum'].dt.year
    df['Month'] = df['Datum'].dt.strftime("%b")
    df['day']   = df['Datum'].dt.strftime("%a")
    # Country extraction
    if 'Location' in df.columns:
        df['Country'] = df['Location'].apply(
            lambda x: x.split(",")[-1].strip() if pd.notnull(x) else 'Unknown'
        )
    else:
        df['Country'] = 'Unknown'
    df['alpha3'] = df['Country']
    # Normalize Rocket costs
    if 'Rocket' not in df.columns:
        df['Rocket'] = 0
    else:
        df['Rocket'] = pd.to_numeric(df['Rocket'], errors='coerce').fillna(0)
    return df

df = load_data()
budget = df[df['Rocket'] > 0].copy()


# 1. Raw Data Overview
st.subheader("1. Raw Data Overview")
st.dataframe(df.head())

# 2. Missing Data
st.subheader("2. Missing Data Percentage")
missed = pd.DataFrame({
    'column': df.columns,
    'missing_percent': (df.isnull().mean() * 100).round(2)
})
st.dataframe(missed)

# 3. Launches by Year
st.subheader("3. Launches by Year")
fig, ax = plt.subplots(figsize=(8,6))
sns.countplot(y=df['year'], palette="viridis", ax=ax)
ax.set(title="Launches by Year", xlabel="Count", ylabel="Year")
st.pyplot(fig)

# 4. Launches by Month
st.subheader("4. Launches by Month")
fig, ax = plt.subplots(figsize=(8,6))
sns.countplot(x='Month', data=df, palette="viridis", ax=ax)
ax.set(title="Launches by Month", xlabel="Month", ylabel="Count")
ax.tick_params(axis='x', rotation=45)
st.pyplot(fig)

# 5. Launches by Day of Week
st.subheader("5. Launches by Day of Week")
days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
df_days = df['day'].value_counts().reindex(days).reset_index()
df_days.columns = ['day','count']
fig, ax = plt.subplots(figsize=(8,4))
sns.barplot(x='day', y='count', data=df_days, ax=ax)
ax.set(title="Launches by Day", xlabel="Day", ylabel="Count")
st.pyplot(fig)

# 6. Rockets Launched per Month
st.subheader("6. Rockets Launched per Month")
df_mon = df['Month'].value_counts().reindex(list(month_abbr)[1:]).reset_index()
df_mon.columns = ['Month','count']
fig, ax = plt.subplots(figsize=(10,5))
bar = sns.barplot(x='Month', y='count', data=df_mon, ax=ax)
for p in bar.patches:
    ax.annotate(int(p.get_height()), (p.get_x()+p.get_width()/2, p.get_height()),
                ha='center', va='bottom')
ax.set(title="Rockets Launched per Month", ylabel="Count")
ax.tick_params(axis='x', rotation=45)
st.pyplot(fig)

# 7. Distribution of Rocket Costs by Status
st.subheader("7. Distribution of Rocket Costs by Status")
if 'Status Rocket' in df.columns:
    dfd = budget  # show all budget entries
    if not dfd.empty:
        fig, ax = plt.subplots(figsize=(10,5))
        sns.histplot(data=dfd, x='Rocket', hue='Status Rocket', bins=20, alpha=0.5, ax=ax)
        ax.set(title="Rocket Cost Distribution by Status", xlabel="Cost", ylabel="Frequency")
        st.pyplot(fig)
    else:
        st.info("No budget entries to plot rocket costs.")

# 8. Successful Missions by Company
st.subheader("8. Successful Missions by Company")
su = df[df['Status Mission']=="Success"]\
      .groupby('Company Name')['Detail'].count()\
      .nlargest(20).reset_index()
fig, ax = plt.subplots(figsize=(12,5))
sns.barplot(x='Company Name', y='Detail', data=su, ax=ax)
ax.set(title="Top 20 Companies by Successful Missions", xlabel="Company", ylabel="Count")
ax.tick_params(axis='x', rotation=45)
st.pyplot(fig)

# 9. Unsuccessful Missions by Company
st.subheader("9. Unsuccessful Missions by Company")
fu = df[df['Status Mission']!="Success"]\
      .groupby('Company Name')['Detail'].count()\
      .nlargest(20).reset_index()
fig, ax = plt.subplots(figsize=(12,5))
sns.barplot(x='Company Name', y='Detail', data=fu, ax=ax)
ax.set(title="Top 20 Companies by Failed Missions", xlabel="Company", ylabel="Count")
ax.tick_params(axis='x', rotation=45)
st.pyplot(fig)

# 10. Total Budget Spent by Company
st.subheader("10. Total Budget Spent by Company")
b = df.groupby('Company Name')['Rocket'].sum().reset_index().sort_values('Rocket', ascending=False)
if not b.empty:
    fig, ax = plt.subplots(figsize=(12,4))
    sns.barplot(x='Company Name', y='Rocket', data=b, ax=ax)
    ax.set(title="Budget by Company", xlabel="Company", ylabel="Cost")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
else:
    st.info("No budget data to display.")

# 11. Company-Year Heatmap
st.subheader("11. Company‑Year Launch Heatmap")
hy = df.groupby(['Company Name','year'])['Detail'].count().reset_index()
topc = hy['Company Name'].value_counts().nlargest(20).index
hm_df = hy[hy['Company Name'].isin(topc)]
hm = hm_df.pivot(index='Company Name', columns='year', values='Detail').fillna(0)
fig = go.Figure(go.Heatmap(
    z=hm.values, x=hm.columns.astype(str), y=hm.index, colorscale='Viridis'
))
fig.update_layout(title="Company‑Year Launch Heatmap", xaxis_title="Year", yaxis_title="Company")
st.plotly_chart(fig)

# 12. Missions Count & Spending by Country
st.subheader("12. Missions Count & Spending by Country")
cnt = df.groupby('Country')['Detail'].count().reset_index(name='count')
spd = df.groupby('Country')['Rocket'].sum().reset_index(name='spend')
cs = cnt.merge(spd, on='Country').sort_values('count', ascending=False)
if not cs.empty:
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                        subplot_titles=("Count","Spend"))
    fig.add_trace(go.Bar(x=cs['count'], y=cs['Country'], orientation='h'), row=1, col=1)
    fig.add_trace(go.Bar(x=cs['spend'], y=cs['Country'], orientation='h'), row=1, col=2)
    fig.update_layout(height=600, title="Country Missions & Spending")
    st.plotly_chart(fig)
else:
    st.info("No country spending data.")

# 13. Pie Chart of Launches by Country
st.subheader("13. Launch Distribution by Country")
pc = df['Country'].value_counts().reset_index()
pc.columns = ['country','count']
st.plotly_chart(px.pie(pc, names='country', values='count', title="Launches by Country"))

# 14. Monthly Launch Time Series
st.subheader("14. Monthly Launches Time Series")
ts = df.set_index('Datum').resample('M')['Detail'].count().reset_index()
ts['month'] = ts['Datum'].dt.to_period('M').dt.to_timestamp()
st.plotly_chart(px.line(ts, x='month', y='Detail', title="Monthly Launches"))

# 15. Seasonal Decomposition
st.subheader("15. Seasonal Decomposition")
decomp = seasonal_decompose(ts.set_index('month')['Detail'], model='additive', period=12)
fig = decomp.plot()
fig.set_size_inches(18, 10)
st.pyplot(fig)

# 16. ARIMA Model for CASC Annual Launches
st.subheader("16. ARIMA Model for CASC")
cas = df[df['Company Name']=='CASC']\
       .groupby('year')['Detail'].count().reset_index(name='launches')
cas = cas[cas['year'] < 2020]
st.plotly_chart(px.line(cas, x='year', y='launches', title="CASC Launches"))
model = ARIMA(cas['launches'], order=(2,2,1)).fit()
# Actual vs Fitted
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(cas['launches'], label='Actual')
ax.plot(model.fittedvalues, label='Fitted')
ax.legend(); ax.set(title="Actual vs Fitted (CASC ARIMA)")
st.pyplot(fig)

# 17. CASC Future Years Forecast
st.subheader("17. CASC Future Years Forecast")
fut = model.get_forecast(steps=6)
fdf = pd.DataFrame({'year': range(2020, 2026), 'forecast': fut.predicted_mean.astype(int)})
all_cas = pd.concat([cas, fdf], ignore_index=True)
st.plotly_chart(px.line(
    all_cas, x='year', y=['launches','forecast'],
    labels={'value':'Launches','variable':'Series'},
    title="Actual vs Forecast (CASC)"
))

# 18. Monthly Launch Forecast (Next 16 Months)
st.subheader("18. Next 16 Months Launch Forecast")
mf = model.get_forecast(steps=16).predicted_mean.astype(int)
months = pd.date_range(start=ts['month'].max()+pd.offsets.MonthBegin(1),
                       periods=16, freq='MS')
m_df = pd.DataFrame({'month': months, 'forecast': mf})
st.plotly_chart(px.line(m_df, x='month', y='forecast', title="Next 16 Months Forecast"))

# 19. Word Cloud of Mission Descriptions
st.subheader("19. Word Cloud of Mission Descriptions")
if 'Detail' in df.columns:
    text = " ".join(df['Detail'].dropna().astype(str))
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    st.image(wc.to_array(), use_container_width=True)
else:
    st.info("No 'Detail' column found; skipping word cloud.")