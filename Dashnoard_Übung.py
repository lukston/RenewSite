import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime

st.set_page_config(layout="wide")
st.title("📈 Yield Curve Dashboard (3D + 2D)")

# File uploader
uploaded_file = st.file_uploader("📂 Upload Excel File with Yield Curve Data", type=["xlsx"])

if uploaded_file:
    # Read Excel file
    df = pd.read_excel(uploaded_file, sheet_name=0)
    
    # Parse and sort date
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.sort_values('DATE')

    # Convert min/max date for slider compatibility
    min_date = df['DATE'].min().date()
    max_date = df['DATE'].max().date()

    # Date range selector for 3D plot
    st.subheader("📊 3D Yield Curve Over Time")
    start_date, end_date = st.slider(
        "Select Time Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )

    # Filter dataframe
    filtered_df = df[(df['DATE'] >= pd.to_datetime(start_date)) & 
                     (df['DATE'] <= pd.to_datetime(end_date))]

    if filtered_df.empty:
        st.warning("⚠️ No data in selected time range.")
    else:
        # --- 3D PLOT ---
        maturities = df.columns[1:]
        dates = filtered_df['DATE'].dt.strftime('%Y-%m-%d')

        X = np.array([maturities] * len(filtered_df))
        Y = np.array([dates]).T
        Z = filtered_df[maturities].values

        fig_3d = go.Figure(data=[go.Surface(
            z=Z,
            x=[str(m) for m in maturities],
            y=dates,
            colorscale='Viridis'
        )])

        fig_3d.update_layout(
            title='Yield Curve Over Time (3D)',
            scene=dict(
                xaxis_title='Maturity',
                yaxis_title='Date',
                zaxis_title='Yield (%)'
            ),
            autosize=True,
            height=800
        )

        st.plotly_chart(fig_3d, use_container_width=True)

        # --- 2D PLOT ---
        st.subheader("📈 Yield Curve for a Specific Date")

        # Second slider to choose single date
        selectable_dates = filtered_df['DATE'].dt.date.unique()
        selected_date = st.select_slider(
            "Select Date",
            options=selectable_dates,
            value=selectable_dates[0]
        )

        # Get data for selected date
        curve = filtered_df[filtered_df['DATE'].dt.date == selected_date]
        if not curve.empty:
            y_values = curve.iloc[0][maturities].values.astype(float)
            x_values = [str(m) for m in maturities]

            fig_2d = go.Figure()
            fig_2d.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                mode='lines+markers',
                name=str(selected_date)
            ))

            fig_2d.update_layout(
                title=f'Yield Curve on {selected_date}',
                xaxis_title='Maturity',
                yaxis_title='Yield (%)',
                height=500
            )

            st.plotly_chart(fig_2d, use_container_width=True)
        else:
            st.warning("No data for selected date.")
else:
    st.info("⬆️ Please upload an Excel file with yield curve data.")