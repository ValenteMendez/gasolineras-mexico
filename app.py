import streamlit as st
import pandas as pd
import plotly.express as px
import json
from pathlib import Path

from utils import (
    load_data,
    prepare_station_data,
    prepare_price_data,
    prepare_volume_data,
    scatter_population_vs_stations,
    bar_chart_stations_by_state,
    bar_chart_top_municipalities,
    bar_chart_stations_per_municipality,
    display_national_avg_prices,
    display_state_price_triplet,
    display_state_price_deviation_triplet,
    display_municipality_price_triplet,
    display_municipality_price_deviation_triplet,
    boxplot_price_distribution_by_state,
    histogram_prices_by_type_and_state,
    product_availability_stats,
    volume_analysis_charts,
    historical_volume_chart
)

# Configuration
DATA_DIR = Path("data")
ANALYSIS_RESULTS_FILE = DATA_DIR / "analysis_results.json"

def load_analysis_results():
    """Load pre-computed analysis results if available."""
    try:
        with open(ANALYSIS_RESULTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def main():
    st.set_page_config(page_title="Gasoline MX Dashboard", page_icon="⛽", layout="wide")
    st.title("Comprehensive Analysis of Gasoline Prices and Volumes in Mexico")

    # Disclaimer
    st.markdown(
        "***Disclaimer**: The data for stations and volumens has been extracted from the CRE (Comisión Reguladora de Energía) databases.*"
    )

    # Try to load pre-computed analysis results
    analysis_results = load_analysis_results()
    
    if analysis_results:
        st.success("Using pre-computed analysis results")
        # TODO: Implement visualization using analysis_results
    else:
        # Load data from CSV files
        df_gas, df_pop, df_vol = load_data(
            gas_prices_path=DATA_DIR / "gas_prices_clean.csv",
            population_path=DATA_DIR / "population.csv",
            volumes_path=DATA_DIR / "volumes.csv"
        )

        # Prepare data
        df_station = prepare_station_data(df_gas, df_pop)
        df_price = prepare_price_data(df_station)
        df_volume = prepare_volume_data(df_vol)

        # Create tabs
        tab_stations, tab_prices, tab_volumes, tab_interpretation = st.tabs([
            "Station Analysis", "Price Analysis", "Volume Analysis", "Interpretation"
        ])

        # ---------- Station Analysis ----------
        with tab_stations:
            st.subheader("Population vs. Number of Stations by State")
            fig_scatter = scatter_population_vs_stations(df_station, df_pop)
            st.plotly_chart(fig_scatter, use_container_width=True)

            st.subheader("Number of Stations per State")
            fig_stations_state = bar_chart_stations_by_state(df_station, df_pop)
            st.plotly_chart(fig_stations_state, use_container_width=True)

            st.subheader("Product Availability Statistics")
            product_availability_stats(df_station)

            st.subheader("Top 15 Municipalities by Number of Stations")
            fig_top_mun = bar_chart_top_municipalities(df_station)
            st.plotly_chart(fig_top_mun, use_container_width=True)

            st.subheader("Average Stations per Municipality by State")
            fig_avg_stations = bar_chart_stations_per_municipality(df_station)
            st.plotly_chart(fig_avg_stations, use_container_width=True)

        # ---------- Price Analysis ----------
        with tab_prices:
            st.subheader("National Average Prices by Fuel Type")
            display_national_avg_prices(df_price)

            st.subheader("Average Price per State by Fuel Type")
            display_state_price_triplet(df_price, df_pop)  # 3 side-by-side bar charts

            st.subheader("Price Deviation from National Average by State")
            display_state_price_deviation_triplet(df_price, df_pop)  # 3 side-by-side deviation charts

            st.subheader("Top 15 Municipalities: Highest Average Price by Fuel Type")
            display_municipality_price_triplet(df_price)  # 3 side-by-side bar charts

            st.subheader("Top 15 Municipalities: Price Deviation from National Average")
            display_municipality_price_deviation_triplet(df_price)  # 3 side-by-side deviation charts

            st.subheader("Box Plot: Price Distribution by State")
            figures = boxplot_price_distribution_by_state(df_price)
            for fig in figures:
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Histogram of Prices by Fuel Type and State")
            histogram_prices_by_type_and_state(df_price)

        # ---------- Volume Analysis ----------
        with tab_volumes:
            volume_analysis_charts(df_volume, df_price, df_station, df_pop)
            st.subheader("Historical Volume Analysis")
            hist_fig = historical_volume_chart(df_volume)
            st.plotly_chart(hist_fig, use_container_width=True)

    # ---------- Interpretation ----------
    with tab_interpretation:
        try:
            with open("interpretation.md", "r", encoding="utf-8") as file:
                interpretation_content = file.read()
                st.markdown(interpretation_content)
        except FileNotFoundError:
            st.error("interpretation.md file not found. Please create this file with your interpretation content.")

    # Footer (for all tabs)
    st.markdown("---")
    st.markdown(
        'Made by **[Valentin Mendez](https://personal-landing-page-vm.lovable.app/)** using information from the '
        '**[CRE](https://www.cre.gob.mx/ConsultaPrecios/GasolinasyDiesel/GasolinasyDiesel.html)** (Comisión Reguladora de Energía) and independent research.'
    )

if __name__ == "__main__":
    main()