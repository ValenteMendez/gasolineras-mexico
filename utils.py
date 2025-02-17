import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

# -------------------------------------------------------------------------
# Data Loading & Preparation
# -------------------------------------------------------------------------

def load_data(gas_prices_path, population_path, volumes_path):
    df_gas = pd.read_csv(gas_prices_path)
    df_pop = pd.read_csv(population_path)
    df_vol = pd.read_csv(volumes_path)
    return df_gas, df_pop, df_vol

def prepare_station_data(df_gas, df_pop):
    """
    1) Drop duplicates on place_id.
    2) Merge with df_pop states to ensure all states appear.
    3) Fill with 0 or NaN for missing station info if any.
    """
    df_gas = df_gas.drop_duplicates(subset=["place_id"])

    all_states = df_pop["Entidad Federativa"].unique()
    df_states_only = pd.DataFrame({"state_name": all_states})
    df_states_only = df_states_only.merge(df_gas, on="state_name", how="left")

    for col in ["place_id", "municipality_name", "regular_price", "premium_price", "diesel_price"]:
        if col not in df_states_only.columns:
            df_states_only[col] = np.nan

    return df_states_only

def remove_price_outliers(df, column, lower_percentile=0.1, upper_percentile=99.9, min_price=12, max_price=35):
    """
    Remove price outliers using both statistical methods and business logic:
    1. Remove prices outside 0.1-99.9th percentile range (more lenient)
    2. Remove prices outside realistic range (12-35 pesos)
    """
    df = df.copy()
    
    # Get percentile bounds
    lower_bound = df[column].quantile(lower_percentile/100)
    upper_bound = df[column].quantile(upper_percentile/100)
    
    # Apply both statistical and business logic bounds
    # Take the more lenient bound in each case
    effective_lower = min(lower_bound, min_price)
    effective_upper = max(upper_bound, max_price)
    
    # Create a mask for valid prices
    mask = (df[column] >= effective_lower) & (df[column] <= effective_upper)
    
    # Apply mask and return cleaned data
    df.loc[~mask, column] = np.nan
    return df

def prepare_price_data(df_station):
    """
    Prepare price data:
    1. Convert to numeric
    2. Remove outliers for each fuel type using same bounds
    """
    df = df_station.copy()
    
    # Convert to numeric
    for fuel_col in ["regular_price", "premium_price", "diesel_price"]:
        df[fuel_col] = pd.to_numeric(df[fuel_col], errors="coerce")
    
    # Remove outliers for each fuel type using same bounds
    for fuel_col in ["regular_price", "premium_price", "diesel_price"]:
        df = remove_price_outliers(df, fuel_col)
    
    return df

def prepare_volume_data(df_vol):
    df_vol["Volumen Vendido (litros)"] = pd.to_numeric(df_vol["Volumen Vendido (litros)"], errors="coerce")
    return df_vol

# -------------------------------------------------------------------------
# Station Analysis
# -------------------------------------------------------------------------

def scatter_population_vs_stations(df_station, df_pop):
    """
    Scatter chart with:
      - X-axis: population (formatted as 800K, 1.2M etc)
      - Y-axis: number of stations (no decimals)
      - Text labels for selected states
      - Tooltip: state name first, then population and stations
    """
    # Key states to always show text
    highlight_states = [
        "México", "Ciudad de México", "Jalisco", "Veracruz de Ignacio de la Llave",
        "Puebla", "Chiapas", "Michoacán", "Nuevo León", "Oaxaca", "Guerrero",
        "Guanajuato"
    ]

    stations_per_state = df_station.groupby("state_name")["place_id"].nunique().reset_index()
    stations_per_state.columns = ["state_name", "num_stations"]

    df_merged = pd.merge(
        stations_per_state,
        df_pop[["Entidad Federativa", "2024 population"]],
        left_on="state_name",
        right_on="Entidad Federativa",
        how="left"
    )

    df_merged["2024 population"] = df_merged["2024 population"].replace(",", "", regex=True)
    df_merged["2024 population"] = pd.to_numeric(df_merged["2024 population"], errors="coerce")

    # Create a text_label column that only shows for highlight_states
    df_merged["text_label"] = df_merged.apply(
        lambda row: row["state_name"] if row["state_name"] in highlight_states else "",
        axis=1
    )

    # Custom hover template to format population numbers
    def format_population(value):
        if value >= 1_000_000:
            return f"{value/1_000_000:.1f}M"
        return f"{int(value/1000)}K"
    
    df_merged["formatted_population"] = df_merged["2024 population"].apply(format_population)
    
    # Format number of stations with thousands separator
    df_merged["formatted_stations"] = df_merged["num_stations"].apply(lambda x: f"{int(x):,}")

    fig = px.scatter(
        df_merged,
        x="2024 population",
        y="num_stations",
        text="text_label",
        custom_data=["state_name", "formatted_population", "formatted_stations"],
        #title="Population vs. Number of Stations by State",
        color_discrete_sequence=["#1f77b4"]
    )

    # Move the text labels so they don't overlap
    fig.update_traces(textposition='top center')

    # Update axis labels and formatting
    fig.update_layout(
        xaxis_title="Population",
        yaxis_title="Number of Stations",
        xaxis=dict(tickformat="~s"),  # Format as 800K, 1.2M etc
        yaxis=dict(tickformat=",d"),  # No decimals, with thousands separator
        hoverlabel=dict(namelength=-1)  # Show full state names
    )

    # Update hover template to show the data in the desired format
    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>" +  # state name
            "Population: %{customdata[1]}<br>" +  # formatted population
            "Number of stations: %{customdata[2]}<extra></extra>"  # formatted stations count
        )
    )
    
    return fig

def bar_chart_stations_by_state(df_station, df_pop):
    """
    Horizontal bar chart of number of stations per state, ensuring all states,
    with a taller layout so labels fit. Uses thousand separators and no decimals in hover.
    """
    stations_per_state = df_station.groupby("state_name")["place_id"].nunique().reset_index()
    stations_per_state.columns = ["state_name", "num_stations"]

    all_states = df_pop["Entidad Federativa"].unique()
    df_all_states = pd.DataFrame({"state_name": all_states})
    df_merge = pd.merge(df_all_states, stations_per_state, on="state_name", how="left")
    df_merge["num_stations"] = df_merge["num_stations"].fillna(0)

    # Format number of stations with thousands separator
    df_merge["formatted_stations"] = df_merge["num_stations"].apply(lambda x: f"{int(x):,}")

    df_merge = df_merge.sort_values("num_stations", ascending=True)

    fig = px.bar(
        df_merge,
        x="num_stations",
        y="state_name",
        orientation="h",
        #title="Number of Stations by State",
        custom_data=["formatted_stations"],
        color_discrete_sequence=["#1f77b4"]
    )

    # Update hover template to show formatted number
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>Number of stations: %{customdata[0]}<extra></extra>"
    )

    # Make it taller for all 32 states
    fig.update_layout(height=900)
    return fig

def product_availability_stats(df_station):
    """
    Display total stations and availability statistics for each fuel type,
    with proper formatting and percentages.
    """
    total_stations = df_station["place_id"].nunique()
    reg_stations = df_station.dropna(subset=["regular_price"])["place_id"].nunique()
    prem_stations = df_station.dropna(subset=["premium_price"])["place_id"].nunique()
    diesel_stations = df_station.dropna(subset=["diesel_price"])["place_id"].nunique()

    # Calculate percentages
    reg_pct = (reg_stations / total_stations) * 100
    prem_pct = (prem_stations / total_stations) * 100
    diesel_pct = (diesel_stations / total_stations) * 100

    st.write(f"**Total Stations: {total_stations:,}**; out of which, stations selling:")
    st.write(f"- Regular: {reg_stations:,} ({reg_pct:.1f}% coverage)")
    st.write(f"- Premium: {prem_stations:,} ({prem_pct:.1f}% coverage)")
    st.write(f"- Diesel: {diesel_stations:,} ({diesel_pct:.1f}% coverage)")

def bar_chart_top_municipalities(df_station, top_n=15):
    """
    Horizontal bar chart of top N municipalities by station count.
    Tooltip includes state name and number of stations formatted with thousand separators, no decimals.
    """
    # We group by BOTH municipality_name and state_name
    group_cols = ["municipality_name", "state_name"]
    stations_by_mun = df_station.groupby(group_cols)["place_id"].nunique().reset_index()
    stations_by_mun.columns = ["municipality_name", "state_name", "num_stations"]

    # Sort descending, pick top N
    stations_by_mun = stations_by_mun.sort_values("num_stations", ascending=False).head(top_n)
    # Then ascending for horizontal
    stations_by_mun = stations_by_mun.sort_values("num_stations", ascending=True)

    # Format number of stations with thousands separator
    stations_by_mun["formatted_stations"] = stations_by_mun["num_stations"].apply(lambda x: f"{int(x):,}")

    fig = px.bar(
        stations_by_mun,
        x="num_stations",
        y="municipality_name",
        orientation="h",
        custom_data=["state_name", "formatted_stations"],
        color_discrete_sequence=["#1f77b4"]
    )

    # Update hover template to show formatted number
    fig.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>" +
            "State: %{customdata[0]}<br>" +
            "Number of stations: %{customdata[1]}<extra></extra>"
        )
    )

    fig.update_layout(height=600)
    return fig

def bar_chart_stations_per_municipality(df_station):
    """
    Horizontal bar chart showing the average number of stations per municipality in each state.
    Calculated as total stations in state / number of municipalities in that state.
    """
    # Count unique municipalities and stations per state
    mun_per_state = df_station.groupby("state_name")["municipality_name"].nunique().reset_index()
    stations_per_state = df_station.groupby("state_name")["place_id"].nunique().reset_index()
    
    # Merge the counts
    df_merged = pd.merge(stations_per_state, mun_per_state, on="state_name")
    df_merged.columns = ["state_name", "total_stations", "total_municipalities"]
    
    # Calculate average
    df_merged["avg_stations_per_municipality"] = df_merged["total_stations"] / df_merged["total_municipalities"]
    
    # Format the numbers for tooltip
    df_merged["formatted_avg"] = df_merged["avg_stations_per_municipality"].apply(lambda x: f"{x:.1f}")
    df_merged["formatted_total"] = df_merged["total_stations"].apply(lambda x: f"{int(x):,}")
    df_merged["formatted_mun"] = df_merged["total_municipalities"].apply(lambda x: f"{int(x):,}")
    
    # Sort by average
    df_merged = df_merged.sort_values("avg_stations_per_municipality", ascending=True)
    
    fig = px.bar(
        df_merged,
        x="avg_stations_per_municipality",
        y="state_name",
        orientation="h",
        custom_data=["formatted_avg", "formatted_total", "formatted_mun"],
        color_discrete_sequence=["#1f77b4"]
    )
    
    # Update hover template to show all relevant information
    fig.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>" +
            "Average stations per municipality: %{customdata[0]}<br>" +
            "Total stations: %{customdata[1]}<br>" +
            "Number of municipalities: %{customdata[2]}<extra></extra>"
        )
    )
    
    # Format x-axis to show one decimal place
    fig.update_layout(
        xaxis_title="Average Number of Stations per Municipality",
        xaxis=dict(tickformat=".1f"),
        height=900
    )
    
    return fig

# -------------------------------------------------------------------------
# Price Analysis
# -------------------------------------------------------------------------

def display_national_avg_prices(df_price):
    avg_regular = df_price["regular_price"].mean()
    avg_premium = df_price["premium_price"].mean()
    avg_diesel = df_price["diesel_price"].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Regular (Avg)", f"${avg_regular:.2f} MXN")
    col2.metric("Premium (Avg)", f"${avg_premium:.2f} MXN")
    col3.metric("Diesel (Avg)", f"${avg_diesel:.2f} MXN")

def display_state_price_triplet(df_price, df_pop):
    """
    3 side-by-side bar charts of avg price by state for Regular (green),
    Premium (red), Diesel (darkgrey), sorted ascending, ensuring 2 decimals in hover.
    """
    fuel_map = {
        "regular_price": ("Regular", "green"),
        "premium_price": ("Premium", "red"),
        "diesel_price": ("Diesel", "darkgrey")
    }
    col1, col2, col3 = st.columns(3)
    all_states = df_pop["Entidad Federativa"].unique()
    df_all_states = pd.DataFrame({"state_name": all_states})

    fuels = list(fuel_map.keys())
    for idx, fuel in enumerate(fuels):
        df_state = df_price.groupby("state_name")[fuel].mean().reset_index()
        df_state.columns = ["state_name", "average_price"]
        df_merged = pd.merge(df_all_states, df_state, on="state_name", how="left")
        df_merged["average_price"] = df_merged["average_price"].fillna(0)

        df_merged = df_merged.sort_values("average_price", ascending=True)
        color_to_use = [fuel_map[fuel][1]]

        fig = px.bar(
            df_merged,
            x="average_price",
            y="state_name",
            orientation="h",
            title=f"Average {fuel_map[fuel][0]} Price by State",
            hover_data={"average_price": ':.2f', "state_name": False},
            color_discrete_sequence=color_to_use
        )
        fig.update_layout(height=700)

        if idx == 0:
            col1.plotly_chart(fig, use_container_width=True)
        elif idx == 1:
            col2.plotly_chart(fig, use_container_width=True)
        else:
            col3.plotly_chart(fig, use_container_width=True)

def display_municipality_price_triplet(df_price):
    """
    3 side-by-side bar charts for the top 15 municipalities by average price
    for Regular (green), Premium (red), Diesel (darkgrey).
    Hover includes municipality, state, average price, deviation and percentage (2 decimals).
    """
    # We need municipality + state grouping to show the state in hover
    fuel_map = {
        "regular_price": ("Regular", "green"),
        "premium_price": ("Premium", "red"),
        "diesel_price": ("Diesel", "darkgrey")
    }
    col1, col2, col3 = st.columns(3)

    fuels = list(fuel_map.keys())
    for idx, fuel in enumerate(fuels):
        # Calculate national average
        national_avg = df_price[fuel].mean()
        
        group_cols = ["municipality_name", "state_name"]
        df_mun = df_price.groupby(group_cols)[fuel].mean().reset_index()
        df_mun.columns = ["municipality_name", "state_name", "average_price"]

        # Calculate deviation and percentage
        df_mun["price_deviation"] = df_mun["average_price"] - national_avg
        df_mun["deviation_pct"] = (df_mun["price_deviation"] / national_avg) * 100
        
        # Format numbers for tooltip
        df_mun["formatted_price"] = df_mun["average_price"].apply(lambda x: f"${x:.2f}")
        df_mun["formatted_deviation"] = df_mun["price_deviation"].apply(lambda x: f"{'+' if x > 0 else ''}{x:.2f}")
        df_mun["formatted_pct"] = df_mun["deviation_pct"].apply(lambda x: f"{'+' if x > 0 else ''}{x:.1f}%")

        # Sort descending by price, pick top 15
        df_mun = df_mun.sort_values("average_price", ascending=False).head(15)
        # Then ascending for bar orientation
        df_mun = df_mun.sort_values("average_price", ascending=True)

        fig = px.bar(
            df_mun,
            x="average_price",
            y="municipality_name",
            orientation="h",
            title=f"{fuel_map[fuel][0]} Price (Avg: ${national_avg:.2f})",
            custom_data=["state_name", "formatted_price", "formatted_deviation", "formatted_pct"],
            color_discrete_sequence=[fuel_map[fuel][1]]
        )

        # Update hover template to show all information
        fig.update_traces(
            hovertemplate=(
                "<b>%{y}</b><br>" +
                "State: %{customdata[0]}<br>" +
                "Price: %{customdata[1]}<br>" +
                "Deviation: %{customdata[2]}<br>" +
                "Percentage: %{customdata[3]}<extra></extra>"
            )
        )
        
        fig.update_layout(height=700)

        if idx == 0:
            col1.plotly_chart(fig, use_container_width=True)
        elif idx == 1:
            col2.plotly_chart(fig, use_container_width=True)
        else:
            col3.plotly_chart(fig, use_container_width=True)

def boxplot_price_distribution_by_state(df_price):
    """
    Three box plots of price distribution by state, one for each fuel type.
    Returns a list of three figures, one for each fuel type.
    No outliers displayed (points=None).
    2-decimal numeric formatting done via y-axis tickformat.
    Consistent colors: Regular (green), Premium (red), Diesel (darkgrey).
    """
    fuel_map = {
        "regular_price": ("Regular", "green"),
        "premium_price": ("Premium", "red"),
        "diesel_price": ("Diesel", "darkgrey")
    }
    
    figures = []
    for fuel, (fuel_name, color) in fuel_map.items():
        # Create box plot for each fuel type
        valid_df = df_price.dropna(subset=[fuel])
        
        fig = px.box(
            valid_df,
            x="state_name",
            y=fuel,
            title=f"{fuel_name} Price Distribution by State",
            points=None,  # Hide outliers
            color_discrete_sequence=[color]
        )
        
        # Update layout
        fig.update_layout(
            xaxis=dict(
                type='category',
                tickangle=45,
                title=""
            ),
            yaxis=dict(
                tickformat=".2f",
                title=f"{fuel_name} Price (pesos)"
            ),
            height=500,  # Slightly shorter since we're stacking
            showlegend=False,
            margin=dict(b=100)  # Add more bottom margin for rotated labels
        )
        
        # Update hover template
        fig.update_traces(
            hovertemplate=(
                "<b>%{x}</b><br>" +
                f"{fuel_name} Price: $%{{y:.2f}}<br>" +
                "<extra></extra>"
            )
        )
        
        figures.append(fig)
    
    return figures

def histogram_prices_by_type_and_state(df_price):
    """
    Histograms for each fuel type showing the distribution of prices.
    - X-axis: price with 2 decimal places
    - Y-axis: number of stations
    - One color per fuel type
    - State filter dropdown affecting all three histograms
    - Hover shows price range and count of stations
    """
    fuel_map = {
        "regular_price": ("Regular", "green"),
        "premium_price": ("Premium", "red"),
        "diesel_price": ("Diesel", "darkgrey")
    }
    
    # Add state selector
    all_states = sorted(df_price["state_name"].unique())
    selected_state = st.selectbox("Select State (affects all histograms)", 
                                ["All States"] + list(all_states))
    
    # Filter data based on state selection
    if selected_state != "All States":
        df_filtered = df_price[df_price["state_name"] == selected_state]
    else:
        df_filtered = df_price
    
    for fuel, (fuel_name, color) in fuel_map.items():
        valid_df = df_filtered.dropna(subset=[fuel])
        
        # Skip if no data available
        if len(valid_df) == 0:
            st.warning(f"No {fuel_name} price data available for {selected_state}")
            continue
        
        state_text = f"in {selected_state}" if selected_state != "All States" else "Across All States"
        mean_price = valid_df[fuel].mean()
        
        fig = px.histogram(
            valid_df,
            x=fuel,
            nbins=50,  # More bins for finer granularity
            title=f"Distribution of {fuel_name} Prices {state_text} (Mean: ${mean_price:.2f} MXN)",
            color_discrete_sequence=[color]
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title=f"{fuel_name} Price ($ MXN)",
            yaxis_title="Number of Stations",
            xaxis=dict(tickformat=".2f"),
            showlegend=False,
            height=500
        )
        
        # Update hover template
        fig.update_traces(
            hovertemplate=(
                "Price Range: $%{x:.2f} MXN<br>" +
                "Number of Stations: %{y}<br>" +
                "<extra></extra>"
            )
        )
        
        # Add mean line with annotation only if we have valid histogram data
        try:
            if (fig.data and 
                hasattr(fig.data[0], 'y') and 
                fig.data[0].y is not None and 
                any(y > 0 for y in fig.data[0].y)):
                
                y_max = max(fig.data[0].y)
                fig.add_vline(x=mean_price, line_dash="dash", line_color="gray")
                fig.add_annotation(
                    x=mean_price,
                    y=y_max,
                    text=f"Mean: ${mean_price:.2f} MXN",
                    showarrow=True,
                    arrowhead=1,
                    yshift=10
                )
        except (AttributeError, IndexError, TypeError):
            # If any error occurs while trying to add the mean line, just skip it
            pass
        
        st.plotly_chart(fig, use_container_width=True)

def display_state_price_deviation_triplet(df_price, df_pop):
    """
    3 side-by-side bar charts showing price deviation from national average for each fuel type.
    Positive deviations in red, negative in green.
    """
    fuel_map = {
        "regular_price": "Regular",
        "premium_price": "Premium",
        "diesel_price": "Diesel"
    }
    
    col1, col2, col3 = st.columns(3)
    all_states = df_pop["Entidad Federativa"].unique()
    df_all_states = pd.DataFrame({"state_name": all_states})

    fuels = list(fuel_map.keys())
    for idx, fuel in enumerate(fuels):
        # Calculate national average
        national_avg = df_price[fuel].mean()
        
        # Calculate state averages
        df_state = df_price.groupby("state_name")[fuel].mean().reset_index()
        df_state.columns = ["state_name", "average_price"]
        
        # Merge with all states
        df_merged = pd.merge(df_all_states, df_state, on="state_name", how="left")
        df_merged["average_price"] = df_merged["average_price"].fillna(0)
        
        # Calculate deviation from national average
        df_merged["price_deviation"] = df_merged["average_price"] - national_avg
        df_merged["deviation_pct"] = (df_merged["price_deviation"] / national_avg) * 100
        
        # Format numbers for tooltip
        df_merged["formatted_price"] = df_merged["average_price"].apply(lambda x: f"${x:.2f}")
        df_merged["formatted_deviation"] = df_merged["price_deviation"].apply(lambda x: f"{'+' if x > 0 else ''}{x:.2f}")
        df_merged["formatted_pct"] = df_merged["deviation_pct"].apply(lambda x: f"{'+' if x > 0 else ''}{x:.1f}%")
        
        # Sort by deviation
        df_merged = df_merged.sort_values("price_deviation", ascending=True)
        
        # Create color array based on deviation
        colors = ['#ff4b4b' if x > 0 else '#2ecc71' for x in df_merged["price_deviation"]]
        
        fig = px.bar(
            df_merged,
            x="price_deviation",
            y="state_name",
            orientation="h",
            title=f"{fuel_map[fuel]} (Avg: ${national_avg:.2f})",
            custom_data=["formatted_price", "formatted_deviation", "formatted_pct"]
        )
        
        # Update bars color based on deviation
        fig.update_traces(
            marker_color=colors,
            hovertemplate=(
                "<b>%{y}</b><br>" +
                "Current price: %{customdata[0]}<br>" +
                "Deviation: %{customdata[1]}<br>" +
                "Percentage: %{customdata[2]}<extra></extra>"
            )
        )
        
        # Update layout
        fig.update_layout(
            height=700,
            xaxis_title="Price Deviation ($)",
            yaxis_title="",
            showlegend=False
        )
        
        # Add a vertical line at x=0
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        
        if idx == 0:
            col1.plotly_chart(fig, use_container_width=True)
        elif idx == 1:
            col2.plotly_chart(fig, use_container_width=True)
        else:
            col3.plotly_chart(fig, use_container_width=True)

def display_municipality_price_deviation_triplet(df_price):
    """
    3 side-by-side bar charts showing price deviation from national average for top 15 municipalities
    by deviation magnitude for each fuel type. Positive deviations in red, negative in green.
    """
    fuel_map = {
        "regular_price": ("Regular", "green"),
        "premium_price": ("Premium", "red"),
        "diesel_price": ("Diesel", "darkgrey")
    }
    col1, col2, col3 = st.columns(3)

    fuels = list(fuel_map.keys())
    for idx, fuel in enumerate(fuels):
        # Calculate national average
        national_avg = df_price[fuel].mean()
        
        # Calculate municipality averages
        group_cols = ["municipality_name", "state_name"]
        df_mun = df_price.groupby(group_cols)[fuel].mean().reset_index()
        df_mun.columns = ["municipality_name", "state_name", "average_price"]
        
        # Calculate deviations
        df_mun["price_deviation"] = df_mun["average_price"] - national_avg
        df_mun["deviation_pct"] = (df_mun["price_deviation"] / national_avg) * 100
        
        # Format numbers for tooltip
        df_mun["formatted_price"] = df_mun["average_price"].apply(lambda x: f"${x:.2f}")
        df_mun["formatted_deviation"] = df_mun["price_deviation"].apply(lambda x: f"{'+' if x > 0 else ''}{x:.2f}")
        df_mun["formatted_pct"] = df_mun["deviation_pct"].apply(lambda x: f"{'+' if x > 0 else ''}{x:.1f}%")
        
        # Get top 15 by absolute deviation
        df_mun["abs_deviation"] = abs(df_mun["price_deviation"])
        df_mun = df_mun.nlargest(15, "abs_deviation")
        df_mun = df_mun.sort_values("price_deviation", ascending=True)
        
        # Create color array based on deviation
        colors = ['#ff4b4b' if x > 0 else '#2ecc71' for x in df_mun["price_deviation"]]
        
        fig = px.bar(
            df_mun,
            x="price_deviation",
            y="municipality_name",
            orientation="h",
            title=f"Price Deviations - {fuel_map[fuel][0]} (Avg: ${national_avg:.2f})",
            custom_data=["state_name", "formatted_price", "formatted_deviation", "formatted_pct"]
        )
        
        # Update bars color based on deviation
        fig.update_traces(
            marker_color=colors,
            hovertemplate=(
                "<b>%{y}</b><br>" +
                "State: %{customdata[0]}<br>" +
                "Current price: %{customdata[1]}<br>" +
                "Deviation: %{customdata[2]}<br>" +
                "Percentage: %{customdata[3]}<extra></extra>"
            )
        )
        
        # Update layout
        fig.update_layout(
            height=700,
            xaxis_title="Price Deviation ($)",
            yaxis_title="",
            showlegend=False
        )
        
        # Add a vertical line at x=0
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        
        if idx == 0:
            col1.plotly_chart(fig, use_container_width=True)
        elif idx == 1:
            col2.plotly_chart(fig, use_container_width=True)
        else:
            col3.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------------------
# Volume Analysis
# -------------------------------------------------------------------------

def volume_analysis_charts(df_volume, df_price, df_station, df_pop):
    """
    Replace tables with charts:
    1) Total Volume by Fuel Type
    2) Total Volume by State & Fuel Type
    3) 2024 Market Value by State
    4) Average Volume per Station by State
    Also includes a national total market value metric.
    """
    # Define consistent colors with more diesel variants
    color_map = {
        "Regular": "#2ecc71",  # green
        "Premium": "#ff4b4b",  # red
        "Diesel": "#333333",  # darkest grey
    }

    # Helper function for consistent number formatting
    def format_volume(x, include_label=True):
        """Format volume in B or M with max 2 decimals"""
        if x >= 1e9:
            val = f"{x/1e9:.2f}B".rstrip('0').rstrip('.')
            return f"{val} liters" if include_label else val
        val = f"{x/1e6:.2f}M".rstrip('0').rstrip('.')
        return f"{val} liters" if include_label else val

    def format_currency(x, include_currency=False, include_usd=False):
        """Format currency in T, B or M with max 2 decimals"""
        # Convert to USD (using 20 MXN to 1 USD rate)
        usd_value = x/20
        
        if x >= 1e12:  # MXN in trillions
            mxn = f"${x/1e12:.2f}T".rstrip('0').rstrip('.')
            if include_usd:
                # Always show USD in billions if less than 1T
                if usd_value < 1e12:
                    usd = f"(USD {usd_value/1e9:.2f}B)".rstrip('0').rstrip('.')
                else:
                    usd = f"(USD {usd_value/1e12:.2f}T)".rstrip('0').rstrip('.')
                return f"{mxn} MXN {usd}" if include_currency else f"{mxn} {usd}"
            return f"{mxn} MXN" if include_currency else mxn
        if x >= 1e9:  # MXN in billions
            mxn = f"${x/1e9:.2f}B".rstrip('0').rstrip('.')
            if include_usd:
                usd = f"(USD {usd_value/1e9:.2f}B)".rstrip('0').rstrip('.')
                return f"{mxn} MXN {usd}" if include_currency else f"{mxn} {usd}"
            return f"{mxn} MXN" if include_currency else mxn
        # MXN in millions
        mxn = f"${x/1e6:.2f}M".rstrip('0').rstrip('.')
        if include_usd:
            usd = f"(USD {usd_value/1e6:.2f}M)".rstrip('0').rstrip('.')
            return f"{mxn} MXN {usd}" if include_currency else f"{mxn} {usd}"
        return f"{mxn} MXN" if include_currency else mxn

    st.subheader("Total Volume by Fuel Type (2024)")
    
    # Create a copy of the volume data and map diesel types to "Diesel"
    df_volume_agg = df_volume[df_volume["Año"] == 2024].copy()
    df_volume_agg["SubProducto"] = df_volume_agg["SubProducto"].replace({
        "Diésel Automotriz": "Diesel",
        "DUBA": "Diesel",
        "Diésel Agricola-Marino": "Diesel"
    })
    
    total_by_fuel = df_volume_agg.groupby("SubProducto")["Volumen Vendido (litros)"].sum().reset_index()
    total_by_fuel = total_by_fuel.sort_values("Volumen Vendido (litros)", ascending=False)
    
    # Calculate total volume and percentages
    total_volume = total_by_fuel["Volumen Vendido (litros)"].sum()
    total_by_fuel["Percentage"] = (total_by_fuel["Volumen Vendido (litros)"] / total_volume) * 100
    
    # Format values in billions/millions and percentages
    total_by_fuel["Formatted Volume"] = total_by_fuel.apply(
        lambda x: f"{format_volume(x['Volumen Vendido (litros)'])} ({x['Percentage']:.1f}%)",
        axis=1
    )
    total_by_fuel["Tooltip"] = total_by_fuel.apply(
        lambda x: f"Volume: {format_volume(x['Volumen Vendido (litros)'])}<br>Share: {x['Percentage']:.1f}%",
        axis=1
    )

    fig_total_by_fuel = px.bar(
        total_by_fuel,
        x="SubProducto",
        y="Volumen Vendido (litros)",
        color="SubProducto",
        color_discrete_map=color_map,
        text="Formatted Volume",
        custom_data=["Tooltip"]
    )
    
    # Total Volume by Fuel Type chart
    fig_total_by_fuel.update_layout(
        yaxis=dict(
            title="Volume (liters)",
            tickformat="~s",
            hoverformat="~s"
        ),
        xaxis_title="Fuel Type",
        showlegend=True
    )
    
    fig_total_by_fuel.update_traces(
        hovertemplate="%{customdata[0]}<extra></extra>"
    )
    
    st.plotly_chart(fig_total_by_fuel, use_container_width=True)

    st.subheader("Total Volume by State & Fuel Type (2024)")
    
    # Add toggle for stacked percentage
    show_percentage = st.checkbox("Show as percentage of state total", value=False, key="volume_percentage")
    
    total_by_state_fuel = df_volume_agg.groupby(["EntidadFederativa", "SubProducto"])["Volumen Vendido (litros)"].sum().reset_index()
    
    # Calculate state totals for sorting
    state_totals = total_by_state_fuel.groupby("EntidadFederativa")["Volumen Vendido (litros)"].sum().reset_index()
    state_order = state_totals.sort_values("Volumen Vendido (litros)", ascending=False)["EntidadFederativa"].tolist()
    
    # Calculate percentage within each state
    total_by_state_fuel = total_by_state_fuel.merge(
        state_totals,
        on="EntidadFederativa",
        suffixes=('', '_state_total')
    )
    total_by_state_fuel['state_percentage'] = (
        total_by_state_fuel["Volumen Vendido (litros)"] / 
        total_by_state_fuel["Volumen Vendido (litros)_state_total"] * 100
    )
    
    # Format values for hover
    total_by_state_fuel["Tooltip"] = total_by_state_fuel.apply(
        lambda x: (
            f"{x['SubProducto']}<br>"
            f"Volume: {format_volume(x['Volumen Vendido (litros)'])}<br>"
            f"Share: {x['state_percentage']:.1f}% of state total"
        ),
        axis=1
    )

    fig_state_fuel = px.bar(
        total_by_state_fuel,
        x="EntidadFederativa",
        y="state_percentage" if show_percentage else "Volumen Vendido (litros)",
        color="SubProducto",
        color_discrete_map=color_map,
        barmode="stack",
        category_orders={"EntidadFederativa": state_order},
        custom_data=["Tooltip"]
    )
    
    # Total Volume by State & Fuel Type chart
    fig_state_fuel.update_layout(
        xaxis=dict(
            title="State",
            type='category',
            tickangle=45
        ),
        yaxis=dict(
            title="Percentage of State Total" if show_percentage else "Volume (liters)",
            tickformat=".1f" if show_percentage else "~s",
            hoverformat=".1f" if show_percentage else "~s"
        ),
        height=700
    )
    
    fig_state_fuel.update_traces(
        hovertemplate="%{customdata[0]}<extra></extra>"
    )
    
    st.plotly_chart(fig_state_fuel, use_container_width=True)

    # Calculate approximate 2024 market values
    avg_reg = df_price["regular_price"].mean()
    avg_prem = df_price["premium_price"].mean()
    avg_diesel = df_price["diesel_price"].mean()

    price_map = {
        "Regular": avg_reg,
        "Premium": avg_prem,
        "Diesel": avg_diesel
    }

    # National total
    volume_2024 = total_by_fuel.copy()
    volume_2024["Avg_Price"] = volume_2024["SubProducto"].map(price_map)
    volume_2024["Market_Value_2024"] = volume_2024["Volumen Vendido (litros)"] * volume_2024["Avg_Price"]
    total_market_value = volume_2024["Market_Value_2024"].sum()
    
    # Calculate percentages of total market value
    volume_2024["Market_Share"] = (volume_2024["Market_Value_2024"] / total_market_value) * 100
    
    # Display total first
    st.subheader("Market Value Analysis (2024)")
    formatted_total = format_currency(total_market_value, include_currency=True, include_usd=True)
    st.metric(label="Total Market Value (All Fuels)", value=formatted_total)
    
    # Format the breakdown data
    volume_2024["Formatted_Volume"] = volume_2024["Volumen Vendido (litros)"].apply(
        lambda x: format_volume(x, include_label=True)
    )
    volume_2024["Formatted_Price"] = volume_2024["Avg_Price"].apply(
        lambda x: f"${x:,.2f} MXN/liter"
    )
    volume_2024["Formatted_Value"] = volume_2024["Market_Value_2024"].apply(
        lambda x: format_currency(x, include_currency=True, include_usd=True)
    )
    
    # Display breakdown for each fuel type
    st.markdown("### Estimated Market Value Breakdown (2024)")
    for _, row in volume_2024.iterrows():
        st.markdown(f"""
        **{row['SubProducto']}** ({row['Market_Share']:.1f}% of total market)
        - Volume: {row['Formatted_Volume']}
        - Average Price: {row['Formatted_Price']}
        - Market Value: {row['Formatted_Value']}
        """)

    # Calculate market value for each state and fuel type
    total_by_state_fuel["Avg_Price"] = total_by_state_fuel["SubProducto"].map(price_map)
    total_by_state_fuel["Market_Value_2024"] = (
        total_by_state_fuel["Volumen Vendido (litros)"] * total_by_state_fuel["Avg_Price"]
    )
    
    # Calculate state totals for sorting
    state_market_totals = total_by_state_fuel.groupby("EntidadFederativa")["Market_Value_2024"].sum().reset_index()
    state_market_order = state_market_totals.sort_values("Market_Value_2024", ascending=False)["EntidadFederativa"].tolist()
    
    # Format market values for hover
    total_by_state_fuel["Formatted Value"] = total_by_state_fuel["Market_Value_2024"].apply(format_currency)
    total_by_state_fuel["Tooltip"] = total_by_state_fuel.apply(
        lambda x: f"{x['SubProducto']}: {format_currency(x['Market_Value_2024'], include_usd=True)}",
        axis=1
    )

    st.subheader("Market Value by State (2024)")
    
    # Add toggle for stacked percentage
    show_percentage = st.checkbox("Show as percentage of state total", value=False)
    
    # Calculate percentages within each state
    state_totals_for_pct = total_by_state_fuel.groupby("EntidadFederativa")["Market_Value_2024"].sum().reset_index()
    total_by_state_fuel = total_by_state_fuel.merge(
        state_totals_for_pct,
        on="EntidadFederativa",
        suffixes=('', '_state_total')
    )
    total_by_state_fuel['state_percentage'] = (
        total_by_state_fuel["Market_Value_2024"] / 
        total_by_state_fuel["Market_Value_2024_state_total"] * 100
    )
    
    # Update tooltip to include percentage
    total_by_state_fuel["Tooltip"] = total_by_state_fuel.apply(
        lambda x: (
            f"{x['SubProducto']}: {format_currency(x['Market_Value_2024'], include_usd=True)}<br>"
            f"({x['state_percentage']:.1f}% of state total)"
        ),
        axis=1
    )

    fig_market_value_by_state = px.bar(
        total_by_state_fuel,
        x="EntidadFederativa",
        y="state_percentage" if show_percentage else "Market_Value_2024",
        color="SubProducto",
        color_discrete_map=color_map,
        barmode="stack",
        category_orders={"EntidadFederativa": state_market_order},
        custom_data=["Tooltip"]
    )
    
    # Market Value chart
    fig_market_value_by_state.update_layout(
        xaxis=dict(
            title="State",
            type='category',
            tickangle=45
        ),
        yaxis=dict(
            title="Percentage of State Total" if show_percentage else "Market Value",
            tickformat=".1f" if show_percentage else "~s",
            hoverformat=".1f" if show_percentage else "~s"
        ),
        height=700
    )
    
    fig_market_value_by_state.update_traces(
        hovertemplate="%{customdata[0]}<extra></extra>"
    )
    
    st.plotly_chart(fig_market_value_by_state, use_container_width=True)

    # Average Volume per Station by State
    st.subheader("Average Volume per Station by State (2024)")
    st.markdown("""
    **Methodology:**
    1. Total volume is calculated as the sum of all fuel types sold in each state in 2024
    2. Number of stations is counted as unique stations (by place_id) in each state
    3. Average = Total Volume / Number of Stations
    4. States with no stations are shown as 0
    """)
    
    total_volume_all = df_volume_agg["Volumen Vendido (litros)"].sum()
    total_stations = df_station["place_id"].nunique()
    avg_vol_per_station = total_volume_all / total_stations if total_stations else np.nan

    formatted_avg = format_volume(avg_vol_per_station)
    st.write(f"**Average Volume per Station (National):** {formatted_avg}")

    stations_per_state = df_station.groupby("state_name")["place_id"].nunique().reset_index()
    stations_per_state.columns = ["EntidadFederativa", "count_stations"]

    vol_by_state = df_volume_agg.groupby("EntidadFederativa")["Volumen Vendido (litros)"].sum().reset_index()
    merged_state_vol = vol_by_state.merge(stations_per_state, on="EntidadFederativa", how="left")
    merged_state_vol["count_stations"] = merged_state_vol["count_stations"].fillna(0)

    merged_state_vol["avg_volume_per_station"] = np.where(
        merged_state_vol["count_stations"] > 0,
        merged_state_vol["Volumen Vendido (litros)"] / merged_state_vol["count_stations"],
        0
    )
    
    # Sort by average volume
    merged_state_vol = merged_state_vol.sort_values("avg_volume_per_station", ascending=True)
    
    # Format for tooltip with additional info
    merged_state_vol["Formatted Average"] = merged_state_vol["avg_volume_per_station"].apply(format_volume)
    merged_state_vol["Tooltip"] = merged_state_vol.apply(
        lambda x: (
            f"Total Volume: {format_volume(x['Volumen Vendido (litros)'])}<br>"
            f"Stations: {int(x['count_stations']):,}<br>"
            f"Average: {format_volume(x['avg_volume_per_station'])}"
        ),
        axis=1
    )

    fig_avg_vol_station = px.bar(
        merged_state_vol,
        x="avg_volume_per_station",
        y="EntidadFederativa",
        orientation="h",
        custom_data=["Tooltip"],
        height=800,
        color_discrete_sequence=["#1e3799"]
    )
    
    # Average Volume per Station chart
    fig_avg_vol_station.update_layout(
        xaxis=dict(
            title="Average Volume per Station (liters)",
            tickformat="~s",
            hoverformat="~s"
        ),
        yaxis_title="State"
    )
    
    fig_avg_vol_station.update_traces(
        hovertemplate="%{customdata[0]}<extra></extra>"
    )
    
    st.plotly_chart(fig_avg_vol_station, use_container_width=True)

    # New scatter plot of volume vs market value
    st.subheader("Volume vs Market Value by State (2024)")
    
    # Prepare data for scatter plot
    scatter_data = state_market_totals.copy()  # Already has EntidadFederativa and Market_Value_2024
    scatter_data = scatter_data.merge(
        state_totals[["EntidadFederativa", "Volumen Vendido (litros)"]], 
        on="EntidadFederativa"
    )
    scatter_data = scatter_data.merge(
        merged_state_vol[["EntidadFederativa", "avg_volume_per_station"]], 
        on="EntidadFederativa"
    )
    
    # Format values for tooltip
    scatter_data["Formatted Volume"] = scatter_data["Volumen Vendido (litros)"].apply(format_volume)
    scatter_data["Formatted Value"] = scatter_data["Market_Value_2024"].apply(
        lambda x: format_currency(x, include_currency=True, include_usd=True)
    )
    scatter_data["Formatted Avg"] = scatter_data["avg_volume_per_station"].apply(format_volume)
    
    fig_scatter = px.scatter(
        scatter_data,
        x="Volumen Vendido (litros)",
        y="Market_Value_2024",
        size="avg_volume_per_station",
        hover_name="EntidadFederativa",
        custom_data=["Formatted Volume", "Formatted Value", "Formatted Avg"]
    )
    
    # Volume vs Market Value scatter plot
    fig_scatter.update_layout(
        xaxis=dict(
            title="Total Volume (liters)",
            tickformat="~s",
            hoverformat="~s"
        ),
        yaxis=dict(
            title="Market Value",
            tickformat="~s",
            hoverformat="~s"
        ),
        height=700
    )
    
    # Update hover template
    fig_scatter.update_traces(
        hovertemplate=(
            "<b>%{hovertext}</b><br>" +
            "Volume: %{customdata[0]}<br>" +
            "Market Value: %{customdata[1]}<br>" +
            "Avg Volume/Station: %{customdata[2]}" +
            "<extra></extra>"
        )
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Volume per Capita Analysis
    st.subheader("Volume per Capita by State")
    st.markdown("""
    **Methodology:**
    1. Total volume is calculated as the sum of all fuel types sold in each state in 2024
    2. Population data is from 2024 projections
    3. Volume per capita = Total Volume / Population
    """)
    
    # Add toggle for showing total vs fuel type breakdown
    show_by_fuel = st.checkbox("Show breakdown by fuel type", value=False, key="per_capita_by_fuel")
    
    # Prepare population data
    df_pop_clean = df_pop.copy()
    df_pop_clean["2024 population"] = df_pop_clean["2024 population"].replace(",", "", regex=True)
    df_pop_clean["2024 population"] = pd.to_numeric(df_pop_clean["2024 population"], errors="coerce")
    
    # Filter for 2024 data only
    df_volume_2024 = df_volume_agg[df_volume_agg["Año"] == 2024].copy()
    
    if show_by_fuel:
        # Calculate volume per capita by fuel type
        per_capita_data = df_volume_2024.groupby(["EntidadFederativa", "SubProducto"])["Volumen Vendido (litros)"].sum().reset_index()
        per_capita_data = per_capita_data.rename(columns={"EntidadFederativa": "state_name"})
        
        per_capita_data = per_capita_data.merge(
            df_pop_clean[["Entidad Federativa", "2024 population"]], 
            left_on="state_name",
            right_on="Entidad Federativa",
            how="left"
        )
        
        per_capita_data["volume_per_capita"] = (
            per_capita_data["Volumen Vendido (litros)"] / per_capita_data["2024 population"]
        )
        
        # Sort by total volume per capita for consistent state ordering
        state_totals = per_capita_data.groupby("state_name")["volume_per_capita"].sum().sort_values(ascending=False)
        state_order = state_totals.index.tolist()
        
        # Format values for hover
        per_capita_data["Formatted Per Capita"] = per_capita_data["volume_per_capita"].apply(
            lambda x: f"{x:,.1f} liters"
        )
        per_capita_data["Formatted Population"] = per_capita_data["2024 population"].apply(
            lambda x: f"{x:,.0f}"
        )
        per_capita_data["Formatted Volume"] = per_capita_data["Volumen Vendido (litros)"].apply(format_volume)
        
        fig_per_capita = px.bar(
            per_capita_data,
            x="volume_per_capita",
            y="state_name",
            color="SubProducto",
            orientation="h",
            custom_data=["Formatted Per Capita", "Formatted Population", "Formatted Volume", "SubProducto"],
            category_orders={"state_name": state_order},
            color_discrete_map=color_map,
            barmode="stack"
        )
        
        # Update hover template for stacked bars
        fig_per_capita.update_traces(
            hovertemplate=(
                "<b>%{y}</b><br>" +
                "%{customdata[3]}<br>" +
                "Per Capita: %{customdata[0]}<br>" +
                "Population: %{customdata[1]}<br>" +
                "Total Volume: %{customdata[2]}" +
                "<extra></extra>"
            )
        )
    else:
        # Calculate total volume per capita
        per_capita_data = df_volume_2024.groupby("EntidadFederativa")["Volumen Vendido (litros)"].sum().reset_index()
        per_capita_data = per_capita_data.rename(columns={"EntidadFederativa": "state_name"})
        
        per_capita_data = per_capita_data.merge(
            df_pop_clean[["Entidad Federativa", "2024 population"]], 
            left_on="state_name",
            right_on="Entidad Federativa",
            how="left"
        )
        
        per_capita_data["volume_per_capita"] = (
            per_capita_data["Volumen Vendido (litros)"] / per_capita_data["2024 population"]
        )
        
        # Sort by volume per capita
        per_capita_data = per_capita_data.sort_values("volume_per_capita", ascending=True)
        
        # Format values for hover
        per_capita_data["Formatted Per Capita"] = per_capita_data["volume_per_capita"].apply(
            lambda x: f"{x:,.1f} liters"
        )
        per_capita_data["Formatted Population"] = per_capita_data["2024 population"].apply(
            lambda x: f"{x:,.0f}"
        )
        per_capita_data["Formatted Volume"] = per_capita_data["Volumen Vendido (litros)"].apply(format_volume)
        
        fig_per_capita = px.bar(
            per_capita_data,
            x="volume_per_capita",
            y="state_name",
            orientation="h",
            custom_data=["Formatted Per Capita", "Formatted Population", "Formatted Volume"],
            color_discrete_sequence=["#1e3799"]  # Dark blue
        )
        
        # Update hover template for single bars
        fig_per_capita.update_traces(
            hovertemplate=(
                "<b>%{y}</b><br>" +
                "Per Capita: %{customdata[0]}<br>" +
                "Population: %{customdata[1]}<br>" +
                "Total Volume: %{customdata[2]}" +
                "<extra></extra>"
            )
        )
    
    # Update layout
    fig_per_capita.update_layout(
        xaxis=dict(
            title="Liters per Capita",
            tickformat=",.1f"
        ),
        yaxis_title="State",
        height=800
    )
    
    st.plotly_chart(fig_per_capita, use_container_width=True)

def historical_volume_chart(df_volume):
    """
    Shows historical volume trends with national view and state selector.
    Includes year-over-year comparison option.
    Returns a tuple of (figure, show_yoy) where show_yoy indicates if YoY view is selected.
    """
    # Filter out future data
    df_filtered = df_volume[df_volume["Año"] != 2025].copy()
    
    # Calculate national totals
    df_national = df_filtered.groupby("Año")["Volumen Vendido (litros)"].sum().reset_index()
    df_national["EntidadFederativa"] = "National Total"
    
    # Prepare state data
    df_states = df_filtered.groupby(["Año", "EntidadFederativa"])["Volumen Vendido (litros)"].sum().reset_index()
    
    # Combine national and state data
    df_combined = pd.concat([df_national, df_states])
    
    # Calculate year-over-year change
    df_yoy = df_combined.copy()
    df_yoy["YoY Change"] = df_yoy.groupby("EntidadFederativa")["Volumen Vendido (litros)"].pct_change() * 100
    
    # UI Controls
    col1, col2 = st.columns(2)
    with col1:
        show_yoy = st.checkbox("Show Year-over-Year Change", value=False)
    
    with col2:
        default_states = ["National Total"]
        all_states = sorted(df_states["EntidadFederativa"].unique())
        selected_states = st.multiselect(
            "Select States to Compare",
            options=["National Total"] + all_states,
            default=default_states
        )
    
    # Filter data based on selection
    df_plot = df_yoy if show_yoy else df_combined
    df_plot = df_plot[df_plot["EntidadFederativa"].isin(selected_states)]

    # Format values for hover using the same format_volume logic
    def format_volume(x, include_label=True):
        """Format volume in B or M with max 2 decimals"""
        if x >= 1e9:
            val = f"{x/1e9:.2f}B".rstrip('0').rstrip('.')
            return f"{val} liters" if include_label else val
        val = f"{x/1e6:.2f}M".rstrip('0').rstrip('.')
        return f"{val} liters" if include_label else val

    df_plot["Formatted Value"] = df_plot.apply(
        lambda x: (
            f"{x['YoY Change']:+,.1f}%" if show_yoy else
            format_volume(x['Volumen Vendido (litros)'])
        ),
        axis=1
    )

    fig = px.line(
        df_plot,
        x="Año",
        y="YoY Change" if show_yoy else "Volumen Vendido (litros)",
        color="EntidadFederativa",
        custom_data=["Formatted Value"]
    )
    
    fig.update_layout(
        xaxis_title="Year",
        yaxis=dict(
            title="Year-over-Year Change (%)" if show_yoy else "Volume (liters)",
            tickformat="+.1f" if show_yoy else "~s",  # Use scientific notation for volume
            hoverformat="+.1f" if show_yoy else "~s"
        )
    )
    
    # Update hover template
    fig.update_traces(
        hovertemplate=(
            "<b>%{x}</b><br>" +
            "%{fullData.name}<br>" +
            ("Change: %{customdata[0]}" if show_yoy else "Volume: %{customdata[0]}") +
            "<extra></extra>"
        )
    )
    
    return fig