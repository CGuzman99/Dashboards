import streamlit as st
import plotly.graph_objects as go
import fastf1
import numpy as np

# Get session data
session = fastf1.get_session(2025, 'Austria', 'R')
session.load(weather=False)

results = session.results[['DriverNumber', 'FullName', 'Abbreviation']]

# Project title
st.title(session.event['OfficialEventName'])

# Select drivers
drivers = st.multiselect("Select drivers",
    results['FullName'],
    default=[],
)

# Filter results of selected drivers
drvs_laps = session.laps.pick_drivers(results.loc[results['FullName'].isin(drivers), 'DriverNumber']).rename(columns={'LapNumber': 'Lap'})

# Create figure
fig = go.Figure()

for drv in drvs_laps['Driver'].unique():
    drv_laps = drvs_laps.loc[drvs_laps['Driver']==drv]
    if drv_laps.empty:
        continue

    df = drv_laps[['Lap', 'Position']]

    # Get driver style (color + linestyle)
    style = fastf1.plotting.get_driver_style(identifier=drv, style=['color', 'linestyle'], session=session)
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
    title='Driver Position by Lap',
    title_x=0.5,
    title_xanchor='center',
    legend_title='Driver',
    hovermode='closest',
    )

# Show on streamlit
st.plotly_chart(fig, use_container_width=True)

###############################################################################################################################

drvs_laps = drvs_laps.pick_quicklaps()
drvs_laps = drvs_laps.reset_index()

# Convert LapTime to seconds
drvs_laps["LapTime(s)"] = drvs_laps["LapTime"].dt.total_seconds()

# Get color mapping
driver_colors = fastf1.plotting.get_driver_color_mapping(session=session)
compound_colors = fastf1.plotting.get_compound_mapping(session=session)

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

compound_order = ["SOFT", "MEDIUM", "HARD"]
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
    title="Lap times distributions",
    title_x=0.5,
    title_xanchor='center',
    xaxis_title="Driver",
    yaxis_title="LapTime (s)",
    template="plotly_white",
    violingap=0,
    violinmode='overlay',
    margin=dict(l=40, r=40, t=40, b=40)
)

# Show on Streamlit
st.plotly_chart(fig, use_container_width=True)

#######################################################################################################

# Get fastest lap per driver
fast_laps = drvs_laps.groupby('Driver').apply(lambda x: x.pick_fastest())


# Sector times
sector_times = {
    'Sector1': {},
    'Sector2': {},
    'Sector3': {}
}

# Get drivers colors
fastf1.plotting.setup_mpl()
driver_colors = fastf1.plotting.DRIVER_COLORS
driver_translate = fastf1.plotting.DRIVER_TRANSLATE

# Map sector times
for idx, lap in fast_laps.iterrows():
    drv = lap['Driver']
    sector_times['Sector1'][drv] = lap['Sector1Time']
    sector_times['Sector2'][drv] = lap['Sector2Time']
    sector_times['Sector3'][drv] = lap['Sector3Time']

# Find fastest driver per sector
fastest_by_sector = {}
for sector in ['Sector1', 'Sector2', 'Sector3']:
    fastest_by_sector[sector] = min(sector_times[sector], key=sector_times[sector].get)

print("Pilotos más rápidos por sector:")
for sector, drv in fastest_by_sector.items():
    print(f"{sector}: {drv}")

# Get the data of the fastest lap overall
overall_fastest_lap = session.laps.pick_fastest()
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
for sector_data, sector_name in zip([sector1, sector2, sector3], ['Sector1', 'Sector2', 'Sector3']):
    distance = sector_data['Distance'].iloc[-1] - sector_data['Distance'].iloc[0]
    drv = fastest_by_sector[sector_name]
    time = fast_laps.loc[fast_laps['Driver']==drv, sector_name+"Time"].dt.total_seconds()
    avg_speed = round((distance*3600)/(time[0]*1000), 2)
    color = driver_colors[driver_translate[drv]]
    customdata = np.stack([[drv]*len(sector_data), [sector_name[-1]]*len(sector_data), [time[0]]*len(sector_data), [avg_speed]*len(sector_data)], axis=-1)
    show_in_legend = drv not in added
    added.add(drv)

    fig.add_trace(go.Scatter(
        x=sector_data['X'],
        y=sector_data['Y'],
        mode='lines',
        line=dict(color=color, width=5),
        name=drv if shown_in_legend else None,
        showlegend=show_in_legend,
        hovertemplate="Piloto: <b>%{customdata[0]}</b><br>Sector: <b>%{customdata[1]}</b><br>Time: <b>%{customdata[2]} s</b><br>Avg. speed: <b>%{customdata[3]} km/h</b><extra></extra>",
        customdata=customdata,
    ))

# Adjust design
fig.update_layout(
    title="Fastest driver per sector in their fastest lap",
    title_x=0.5,
    title_xanchor='center',
    showlegend=True,
    autosize=True,
    xaxis=dict(showgrid=False, zeroline=False, visible=False),
    yaxis=dict(showgrid=False, zeroline=False, visible=False),
)

st.plotly_chart(fig, use_container_width=True)