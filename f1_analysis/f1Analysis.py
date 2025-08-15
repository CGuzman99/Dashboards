import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import fastf1
from fastf1 import plotting
from fastf1.ergast import Ergast
import pandas as pd
import numpy as np

def format_timedelta(td):
    total_seconds = td.total_seconds()
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes}:{seconds:06.3f}"

today = pd.Timestamp.today()
with st.sidebar:
    st.title("Race settings")
    seasons = [i for i in range(2018, today.year+1)]
    season = st.selectbox('Season', seasons, index=len(seasons)-1)
    events = fastf1.get_event_schedule(season)[['EventDate', 'EventName']]
    events = events.loc[events['EventDate'] < today]
    last_event_index = len(events.loc[events['EventDate']<=today, 'EventDate']) - 1
    event = st.selectbox('Grand Prix', events['EventName'], index=last_event_index)

    drivers_lst = Ergast(result_type='pandas').get_driver_standings(season=season).content[0]['driverCode']

    drivers = st.multiselect("Select drivers",
        drivers_lst,
        default=[],
    )

    st.session_state["drivers"] = drivers
# Inicializar la sesión solo una vez
if "data_loaded" not in st.session_state:
    st.session_state["data_loaded"] = False
    st.session_state["data"] = None
    st.session_state["year"] = season
    st.session_state["gp"] = event
    st.session_state["session_type"] = "R"

if (season != st.session_state['year']) or (event != st.session_state['gp']):
    st.session_state["year"] = season
    st.session_state["gp"] = event
    st.session_state["data_loaded"] = False  # Fuerza recarga

# Cargar datos solo si no están cargados
if not st.session_state["data_loaded"]:
    with st.spinner("Loading session data..."):
        session = fastf1.get_session(
            st.session_state["year"],
            st.session_state["gp"],
            st.session_state["session_type"]
        )
        session.load(weather=False)
        st.session_state["data"] = session
        st.session_state["data_loaded"] = True
#######################################################################################################################
results = st.session_state['data'].results[['DriverNumber', 'LastName', 'Abbreviation']]# Select drivers
# Project title
st.title(st.session_state['data'].event['OfficialEventName'])
##########################################################################################################################

st.subheader('Driver Position by Lap')
# Filter results of selected drivers
drvs_laps = st.session_state['data'].laps.pick_drivers(results.loc[results['Abbreviation'].isin(drivers), 'DriverNumber']).rename(columns={'LapNumber': 'Lap'})

# Create figure
fig = go.Figure()

for drv in drvs_laps['Driver'].unique():
    drv_laps = drvs_laps.loc[drvs_laps['Driver']==drv]
    if drv_laps.empty:
        continue

    df = drv_laps[['Lap', 'Position']]

    # Get driver style (color + linestyle)
    style = plotting.get_driver_style(identifier=drv, style=['color', 'linestyle'], session=st.session_state['data'])
    color = style['color']
    dash = 'dash' if style['linestyle']=='dashed' else 'solid'  # 'solid', 'dashed', 'dotted', etc.

    # Add trace
    fig.add_trace(go.Scatter(
        x=df['Lap'],
        y=df['Position'],
        mode='lines',
        name=drv,
        line=dict(color=color, dash=dash),
        hovertemplate='Driver: ' + drv + '<br>Lap: %{x}<br>Position: %{y}<extra></extra>'
    ))

# Invert y axis
fig.update_yaxes(title='Position', autorange='reversed')
fig.update_xaxes(title='Lap')
fig.update_layout(
    #title='Driver Position by Lap',
    #title_x=0.5,
    #title_xanchor='center',
    legend_title='Driver',
    hovermode='closest',
    )

# Show on streamlit
st.plotly_chart(fig, use_container_width=True)

#######################################################################################################

st.subheader('Fastest driver per sector')

# Get fastest lap per driver
fast_laps = drvs_laps.groupby('Driver').apply(lambda x: x.pick_fastest())


# Sector times
sector_times = {
    'Sector1': {},
    'Sector2': {},
    'Sector3': {}
}

# Get drivers colors
plotting.setup_mpl()
driver_colors = plotting.get_driver_color_mapping(session=st.session_state['data'])
driver_translate = plotting.DRIVER_TRANSLATE

# Map sector times
for idx, lap in fast_laps.iterrows():
    drv = lap['Driver']
    sector_times['Sector1'][drv] = lap['Sector1Time']
    sector_times['Sector2'][drv] = lap['Sector2Time']
    sector_times['Sector3'][drv] = lap['Sector3Time']

# Find fastest driver per sector
fastest_by_sector = pd.DataFrame(columns=['Driver', 'Time(s)'])
fastest_by_sector.index.name = 'Sector'
for sector in ['Sector1', 'Sector2', 'Sector3']:
    if not sector_times[sector]:
        break 
    fastest_by_sector.loc[sector[-1], 'Driver'] = min(sector_times[sector], key=sector_times[sector].get)
    fastest_by_sector.at[sector[-1], 'Time(s)'] = fast_laps.loc[fast_laps['Driver']==fastest_by_sector.at[sector[-1], 'Driver'], sector+"Time"][0].total_seconds()

# Get the data of the fastest lap overall
overall_fastest_lap = st.session_state['data'].laps.pick_fastest()
car_data = overall_fastest_lap.get_car_data().add_distance()
lap_telemetry = overall_fastest_lap.get_telemetry().add_distance()


# Calculate sector split points based on time
t0 = car_data['Time'].iloc[0]
s1_time = overall_fastest_lap['Sector1Time']
s2_time = overall_fastest_lap['Sector2Time']
s3_time = overall_fastest_lap['Sector3Time']
t1 = t0 + s1_time
t2 = t1 + s2_time

# Divide the data in sectors
sector1 = lap_telemetry[lap_telemetry['Time'] <= t1]
sector2 = lap_telemetry[(lap_telemetry['Time'] > t1) & (lap_telemetry['Time'] <= t2)]
sector3 = lap_telemetry[lap_telemetry['Time'] > t2]

# Create figure
fig = go.Figure()

# Trazado base (gris)
fig.add_trace(go.Scatter(
    x=lap_telemetry['X'],
    y=lap_telemetry['Y'],
    mode='lines',
    line=dict(color='lightgray', width=2),
    name='Trazada'
))

added = set()
# Add each sector with the color of the fastest driver per sector
for sector_data, sector_name in zip([sector1, sector2, sector3], ['1', '2', '3']):
    if fastest_by_sector.empty:
        break
    distance = sector_data['Distance'].iloc[-1] - sector_data['Distance'].iloc[0]
    drv = fastest_by_sector.loc[sector_name, 'Driver']
    time = fastest_by_sector.loc[sector_name, 'Time(s)']
    avg_speed = round((distance*3600)/(time*1000), 2)
    color = driver_colors.get(drv, 'lightgray')
    customdata = np.stack([[drv]*len(sector_data), [sector_name]*len(sector_data), [time]*len(sector_data), [avg_speed]*len(sector_data)], axis=-1)
    show_in_legend = drv not in added
    added.add(drv)

    fig.add_trace(go.Scatter(
        x=sector_data['X'],
        y=sector_data['Y'],
        mode='lines',
        line=dict(color=color, width=5),
        name=drv if show_in_legend else None,
        showlegend=show_in_legend,
        hovertemplate="Piloto: <b>%{customdata[0]}</b><br>Sector: <b>%{customdata[1]}</b><br>Time: <b>%{customdata[2]} s</b><br>Avg. speed: <b>%{customdata[3]} km/h</b><extra></extra>",
        customdata=customdata,
    ))

# Adjust design
fig.update_layout(
    #title="Fastest driver per sector in their fastest lap",
    #title_x=0.5,
    #title_xanchor='center',
    showlegend=True,
    autosize=True,
    xaxis=dict(showgrid=False, zeroline=False, visible=False),
    yaxis=dict(showgrid=False, zeroline=False, visible=False),
)

tab1, tab2 = st.tabs(['Chart', 'Dataframe'])
tab1.plotly_chart(fig, use_container_width=True)
tab2.dataframe(fastest_by_sector)

###############################################################################################################################

st.subheader("Lap times distributions")

drvs_laps = drvs_laps.loc[(~drvs_laps['TrackStatus'].str.contains('|'.join(['3', '4', '5', '6', '7']))) & (drvs_laps['PitInTime'].isnull()) & (drvs_laps['PitOutTime'].isnull())]
drvs_laps = drvs_laps.reset_index()

# Convert LapTime to seconds
drvs_laps["LapTime(s)"] = drvs_laps["LapTime"].dt.total_seconds()

# Get color mapping
driver_colors = plotting.get_driver_color_mapping(session=st.session_state['data'])
compound_colors = plotting.get_compound_mapping(session=st.session_state['data'])

# Create figure
fig = go.Figure()

# Add a violin plot per driver
for drv in drvs_laps['Driver'].unique():
    laps = drvs_laps[drvs_laps["Driver"] == drv]
    if laps.empty:
        continue

    fig.add_trace(go.Violin(
        x=[drv] * len(laps),
        y=laps["LapTime(s)"],
        name=drv,
        line_color=driver_colors.get(drv, 'gray'),
        fillcolor=driver_colors.get(drv, 'gray'),
        opacity=0.6,
        box_visible=False,
        meanline_visible=False,
        showlegend=False
    ))

compound_order = drvs_laps['Compound'].unique()
shown_in_legend = set()
for compound in compound_order:
    compound_laps = drvs_laps[drvs_laps["Compound"] == compound]
    for drv in drvs_laps['Driver'].unique():
        laps = compound_laps[compound_laps["Driver"] == drv]
        if laps.empty:
            continue

        show = compound not in shown_in_legend
        shown_in_legend.add(compound)
        x_jittered = [drv] * len(laps)
        fig.add_trace(go.Scatter(
            x=x_jittered,
            y=laps["LapTime(s)"],
            mode='markers',
            marker=dict(
                color=compound_colors.get(compound, 'gray'),
                size=5,
                line=dict(width=0)
            ),
            name=compound,
            legendgroup=compound,
            showlegend=show  # Show once per compound
        ))

# Adjust design
fig.update_layout(
    xaxis_title="Driver",
    yaxis_title="LapTime (s)",
    template="plotly_white",
    violingap=0,
    violinmode='overlay',
    legend_title='Compound',
    margin=dict(l=40, r=40, t=40, b=40)
)

# Show on Streamlit
st.plotly_chart(fig, use_container_width=True)

#####################################################################################################################

st.subheader("Driver Stint Comparison")

laps = st.session_state['data'].laps.pick_drivers(results.loc[results['Abbreviation'].isin(drivers), 'DriverNumber'])
driver_colors = plotting.DRIVER_COLORS

stints = pd.DataFrame(columns=['Driver', 'Stint', 'Compound', 'AvgLapTimeFormatted', 'AvgLapTime(s)', 'StdLapTime', 'NumLaps'])

# Get average time per stint
laps_grouped = laps.groupby(['Driver', 'Stint', 'Compound'])
stints['AvgLapTime'] = laps_grouped['LapTime'].mean()
stints['AvgLapTime(s)'] = round(stints['AvgLapTime'].dt.total_seconds(), 3)
stints['AvgLapTimeFormatted'] = stints['AvgLapTime'].fillna(pd.Timedelta(0)).apply(format_timedelta)
stints['StdLapTime'] = laps_grouped['LapTime'].std()
stints['NumLaps'] = laps_grouped['LapNumber'].count()
stints['StdLapTime'] = stints['StdLapTime'].fillna(pd.Timedelta(0)).apply(format_timedelta)

# Create Figure
fig1 = go.Figure()

shown_in_legend = set()
# Create a bar per stint
for row in stints.index:
    show = row[2] not in shown_in_legend
    shown_in_legend.add(row[2])
    fig1.add_trace(go.Bar(
        x=[row[0]],
        y=[stints.loc[row, 'NumLaps']],
        name=row[2],
        offsetgroup=row[1],
        marker_color=compound_colors.get(row[2], '#888888'),
        showlegend=show,  # Show once per compound
        hovertemplate=
            f"<b>Driver:</b> {row[0]}<br>" +
            f"<b>Stint:</b> {row[1]}<br>" +
            f"<b>Compound:</b> {row[2]}<br>" +
            f"<b>Avg Lap Time:</b> {stints.loc[row, 'AvgLapTimeFormatted']}<br>" +
            f"<b>Std Dev:</b> {stints.loc[row, 'StdLapTime']} s<br>" +
            f"<b>Laps:</b> {stints.loc[row, 'NumLaps']}"
    ))

fig1.update_traces(textposition='inside', textfont_size=12)
fig1.update_layout(
    xaxis={'categoryorder': 'category ascending'},
    xaxis_title='Driver',
    yaxis_title='Number of laps',
    barmode='stack',
    legend_title='Compound'
)

# Create figure
fig2 = go.Figure()

shown_in_legend = set()
stints = stints.drop(columns=[col for col in stints.index.names if col in stints.columns])
stints = stints.reset_index()
for drv in laps['Driver'].unique():
    if laps.empty:
        continue
    show = drv not in shown_in_legend
    shown_in_legend.add(drv)

    # Get driver style (color + linestyle)
    style = plotting.get_driver_style(identifier=drv, style=['color', 'linestyle'], session=st.session_state['data'])
    color = style['color']
    dash = 'dash' if style['linestyle']=='dashed' else 'solid'  # 'solid', 'dashed', 'dotted', etc.

    fig2.add_trace(go.Scatter(
        x=stints.loc[stints['Driver']==drv, 'Stint'].astype(int),
        y=stints.loc[stints['Driver']==drv, 'AvgLapTime(s)'],
        mode='lines+markers',
        name=drv,
        line=dict(color=color, dash=dash),
        showlegend=show,  # Show once per compound
        hovertemplate=[
            f"<b>Driver:</b> {drv}<br>" +
            f"<b>Stint:</b> {int(row['Stint'])}<br>" +
            f"<b>Compound:</b> {row['Compound']}<br>" +
            f"<b>Avg Lap Time:</b> {row['AvgLapTimeFormatted']}<br>"
            f"<b>Std Dev:</b> {row['StdLapTime']} s<br>" +
            f"<b>Laps:</b> {row['NumLaps']}"
            for _,row in stints.loc[stints['Driver']==drv].iterrows()
        ],
    ))

#fig2.update_traces(textposition='inside', textfont_size=12)
fig2.update_layout(
    xaxis=dict(
        tickmode='linear',
        tick0=1,         # Valor inicial del eje x
        dtick=1          # Distancia entre ticks (1 = solo enteros consecutivos)
    ),
    xaxis_title='Stint',
    yaxis_title='Avg Lap Time (s)',
    hovermode='closest',
    legend_title='Driver'
)

tab1, tab2, tab3 = st.tabs(["Number of laps", 'Avg lap time', "Dataframe"])
tab1.plotly_chart(fig1)
tab2.plotly_chart(fig2)
tab3.dataframe(stints[['Driver', 'Stint', 'Compound', 'AvgLapTimeFormatted', 'AvgLapTime(s)', 'StdLapTime', 'NumLaps']].rename(columns={'AvgLapTimeFormatted': 'Avg Lap Time', 'AvgLapTime(s)': 'Avg Lap Time (s)', 'StdLapTime': 'Std Dev', 'NumLaps': 'Number of Laps'}))