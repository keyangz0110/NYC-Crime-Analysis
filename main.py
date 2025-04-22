import streamlit as st
import pandas as pd
import plotly.express as px
import pydeck as pdk

#branch

# Load data
df_monthly = pd.read_csv("data/arrest_monthly.csv", names=["month", "arrest_count"], header=None)
df_daily = pd.read_csv("data/arrest_daily.csv", names=["day", "arrest_count"], header=None)
df_top = pd.read_csv("data/arrest_top_offenses.csv", names=["offense", "arrest_count"], header=None)
df_demo = pd.read_csv("data/arrest_demographics.csv", names=["sex", "age_group", "race", "arrest_count"], header=None)

sex_map = {
    "M": "Male",
    "F": "Female",
    "U": "Unknown"
}
boro_map = {
    "K": "Brooklyn",
    "B": "Bronx",
    "M": "Manhattan",
    "Q": "Queens",
    "S": "Staten Island"
}
age_map = {
    "<18": "Under 18",
    "18-24": "18 to 24",
    "25-44": "25 to 44",
    "45-64": "45 to 64",
    "65+": "65 and above",
    "UNKNOWN": "Unknown"
}
df_boro_sex = pd.read_csv("data/arrest_borough_gender.csv", names=["ARREST_BORO", "PERP_SEX", "arrest_count"], header=None)
df_boro_sex["ARREST_BORO"] = df_boro_sex["ARREST_BORO"].map(boro_map)
df_boro_sex["PERP_SEX"] = df_boro_sex["PERP_SEX"].map(sex_map)

df_boro_age = pd.read_csv("data/arrest_borough_age.csv", names=["ARREST_BORO", "AGE_GROUP", "arrest_count"], header=None)
df_boro_age["ARREST_BORO"] = df_boro_age["ARREST_BORO"].map(boro_map)
df_boro_age["AGE_GROUP"] = df_boro_age["AGE_GROUP"].map(age_map)

df_boro_race = pd.read_csv("data/arrest_borough_race.csv", names=["ARREST_BORO", "PERP_RACE", "arrest_count"], header=None)
df_boro_race["ARREST_BORO"] = df_boro_race["ARREST_BORO"].map(boro_map)

df_map = pd.read_csv("data/cleaned_data.csv")
# 正确解析 2025/1/2 格式
df_map['ARREST_DATE'] = pd.to_datetime(df_map['ARREST_DATE'], format='%Y/%m/%d')

# 将时间转换为 Python 原生 datetime 对象
min_date = df_map['ARREST_DATE'].min().to_pydatetime()
max_date = df_map['ARREST_DATE'].max().to_pydatetime()


st.set_page_config(layout="wide")
st.title("🚔 NYC Arrests Interactive Dashboard")

# Sidebar filters
with st.sidebar:
    st.header("🔍 Filters")
    selected_boro = st.selectbox("Select Borough", options=["ALL"] + sorted(df_map['ARREST_BORO'].dropna().unique().tolist()))
    selected_sex = st.selectbox("Select Gender", options=["ALL"] + sorted(df_map['PERP_SEX'].dropna().unique().tolist()))
    # 时间筛选器（Streamlit slider）
    date_range = st.slider(
        "Select Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )

    # 应用时间过滤
    df_map = df_map[
        (df_map['ARREST_DATE'] >= date_range[0]) &
        (df_map['ARREST_DATE'] <= date_range[1])
        ]

    #date_range = st.slider("Select Date Range", min_value=pd.to_datetime(df_map['ARREST_DATE']).min(),
     #                      max_value=pd.to_datetime(df_map['ARREST_DATE']).max(),
      #                     value=(pd.to_datetime(df_map['ARREST_DATE']).min(), pd.to_datetime(df_map['ARREST_DATE']).max()))

# Apply filters to map data
map_data = df_map.copy()
map_data['ARREST_DATE'] = pd.to_datetime(map_data['ARREST_DATE'])
map_data = map_data[(map_data['ARREST_DATE'] >= date_range[0]) & (map_data['ARREST_DATE'] <= date_range[1])]
if selected_boro != "ALL":
    map_data = map_data[map_data['ARREST_BORO'] == selected_boro]
if selected_sex != "ALL":
    map_data = map_data[map_data['PERP_SEX'] == selected_sex]

# Tabs
tabs = st.tabs(["📈 Trends", "👤 Demographics", "🧭 Borough Comparison", "🗺️ Maps", "🔝 Top Offenses"])

with tabs[0]:
    st.subheader("Monthly Arrest Trends")
    fig_month = px.line(df_monthly, x="month", y="arrest_count", markers=True)
    st.plotly_chart(fig_month, use_container_width=True)

    st.subheader("Daily Arrest Trends")
    fig_day = px.line(df_daily, x="day", y="arrest_count")
    st.plotly_chart(fig_day, use_container_width=True)

with tabs[1]:
    st.subheader("Gender Distribution")
    fig_sex = px.bar(df_demo.groupby('sex')['arrest_count'].sum().reset_index(), x="sex", y="arrest_count")
    st.plotly_chart(fig_sex, use_container_width=True)

    st.subheader("Age Group Distribution")
    fig_age = px.bar(df_demo.groupby('age_group')['arrest_count'].sum().reset_index(), x="age_group", y="arrest_count")
    st.plotly_chart(fig_age, use_container_width=True)

    st.subheader("Race Distribution")
    fig_race = px.bar(df_demo.groupby('race')['arrest_count'].sum().reset_index(), x="race", y="arrest_count")
    st.plotly_chart(fig_race, use_container_width=True)

with tabs[2]:
    st.subheader("Borough × Gender")
    fig1 = px.bar(df_boro_sex, x="ARREST_BORO", y="arrest_count", color="PERP_SEX", barmode="group")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Borough × Age")
    fig2 = px.bar(df_boro_age, x="ARREST_BORO", y="arrest_count", color="AGE_GROUP", barmode="group")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Borough × Race")
    fig3 = px.bar(df_boro_race, x="ARREST_BORO", y="arrest_count", color="PERP_RACE", barmode="group")
    st.plotly_chart(fig3, use_container_width=True)

with tabs[3]:
    st.subheader("Hexagon Heatmap")
    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
            latitude=40.7128,
            longitude=-74.0060,
            zoom=10,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                "HexagonLayer",
                data=map_data,
                get_position='[Longitude, Latitude]',
                radius=200,
                elevation_scale=4,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
            ),
        ],
    ), use_container_width=True)  # ✅ 让热图横向撑满

    st.subheader("Arrest Points Clustering")
    fig_map = px.scatter_mapbox(
        map_data.sample(min(1000, len(map_data))),
        lat="Latitude", lon="Longitude",
        hover_name="OFNS_DESC", color="ARREST_BORO", zoom=10
    )
    fig_map.update_layout(
        mapbox_style="open-street-map",
        height=600
    )
    st.plotly_chart(fig_map, use_container_width=True)  # ✅ 聚类图也撑满


with tabs[4]:
    st.subheader("Top Offenses by Arrest Count")
    fig_top = px.bar(df_top.sort_values(by="arrest_count", ascending=False),
                     x="offense", y="arrest_count")
    st.plotly_chart(fig_top, use_container_width=True)
