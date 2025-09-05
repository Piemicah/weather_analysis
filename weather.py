import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import f_oneway
import streamlit as st


excel_file = "ALL DATA EXTRACT UPDATE.xlsx"
crop_file = "2011 - 2024 APS KWADP  ORIGINAL.xlsx"

st.set_page_config(page_title="Crop Yields", layout="wide", page_icon="🍀")

st.title("Kwara State Weather and Crop Yield Analysis 🌾")
st.subheader("By Adedoyin Saheed Oyewole")


class Weather:
    def __init__(self, town):
        self.town = town
        try:
            self.data = pd.read_excel(excel_file, town)
            self.clean_data()
        except Exception as e:
            st.error(f"Error loading weather data for {town}: {e}")
            self.data = None

    def clean_data(self):
        if self.data is not None:
            self.data["Solar Irradiance (MJ/m^2/day)"] = [
                np.nan if x < 0 else x
                for x in self.data["Solar Irradiance (MJ/m^2/day)"]
            ]
            self.data.fillna(
                {
                    "Solar Irradiance (MJ/m^2/day)": self.data[
                        "Solar Irradiance (MJ/m^2/day)"
                    ].mean()
                },
                inplace=True,
            )
            self.data["TIME"] = self.data["TIME"].apply(lambda x: str(x))
            self.data["Year"] = self.data["TIME"].apply(lambda x: x.split("-")[0])
            self.data["Month"] = self.data["TIME"].apply(lambda x: x.split("-")[1])
            self.data.drop(
                columns=["TIME", "LAT", "LON", "LOCATION (STATION)"], inplace=True
            )
            self.months = self.data.groupby(["Year", "Month"]).mean()
            self.yearly = self.data.drop(columns="Month").groupby("Year").mean()


def yield_df(crop_data):
    """
    This function creates a DataFrame of crop yields from 2011-2024.
    """
    try:
        col = crop_data["CROPS"].to_list()[::4][:16]
        idx = crop_data.columns[1:]
        frames = []
        for i in range(3, 64, 4):
            f = crop_data.iloc[i, 1:].to_frame()
            frames.append(f)

        df = pd.concat(frames, axis=1)
        yields_df = pd.DataFrame(df.values, columns=col, index=idx)

        # Replace "-" with 0
        yields_df = yields_df.replace("-", 0)

        # Convert all values to numeric safely
        yields_df = yields_df.apply(pd.to_numeric, errors="coerce").fillna(0)

        yields_df.index = np.arange(2010, 2025)

        yields_df = yields_df.drop(index=2010)

        return yields_df
    except Exception as e:
        st.error(f"Error processing crop data: {e}")
        return None


def anova(data, feature):
    alpha = 0.05
    frames = [data[i][feature] for i in data.columns.get_level_values(0).unique()]
    dat = pd.concat(frames, axis=1).drop("2024")
    dat.columns = ["a", "b", "c", "d"]
    F, p = f_oneway(dat.a, dat.b, dat.c, dat.d)
    st.write(f"**Feature:** {feature}")
    st.write(f"F-statistic = {F:.4f}")
    st.write(f"p-value = {p}")
    if p < alpha:
        st.write(
            "There is sufficient evidence that the four average measurements are not the same."
        )
    else:
        st.write("There is not enough evidence to reject the Null Hypothesis.")
    st.dataframe(dat.head())
    st.markdown("---")


def plot_crop(yields_data, weather_data, crop_name):
    st.subheader(f"Effect of Weather on {crop_name.upper()} Yields")
    model = LinearRegression()
    try:
        x = yields_data.loc[2011:2024, crop_name].values.reshape(-1, 1)
    except KeyError:
        st.error(f"Crop '{crop_name}' not found in the dataset.")
        return

    for feature in weather_data.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        y = weather_data.loc["2011":"2024", feature].values.reshape(-1, 1)
        model.fit(x, y)
        y_pred = model.predict(x)
        ax.plot(x, y_pred, color="r", label="Linear Regression")
        ax.scatter(x, y, color="b", label="Actual Data")
        ax.set_title(f"Effect of {feature} on {crop_name} (2011-2024)")
        ax.set_xlabel(f"{crop_name} (tons/Ha)")
        ax.set_ylabel(feature)
        ax.legend()
        st.pyplot(fig)


# Main Application Logic
try:
    with st.spinner("Loading and processing weather data..."):
        il = Weather("ILORIN")
        omu = Weather("OMU ARAN")
        okuta = Weather("OKUTA")
        patigi = Weather("PATIGI")

    if all(w.data is not None for w in [il, omu, okuta, patigi]):
        data1 = il.yearly.loc["1997":"2024"]
        data2 = omu.yearly.loc["1997":"2024"]
        data3 = okuta.yearly.loc["1997":"2024"]
        data4 = patigi.yearly.loc["1997":"2024"]

        combined_data = pd.concat([data1, data2, data3, data4], axis=1)
        data = combined_data.stack().groupby(level=[0, 1]).mean().unstack()

        st.success("Weather data loaded and processed successfully!")

        st.header("Combined Weather Data for Kwara State (1997-2024)")
        st.dataframe(data)

        # Time Series Plots
        st.header("Yearly Time Series Plots for Kwara State")
        df_kwara = Weather("OMU ARAN")
        df_kwara.yearly = data.copy()

        features_to_plot = df_kwara.yearly.columns
        for feature in features_to_plot:
            years = df_kwara.yearly.index.astype(int)
            fig, ax = plt.subplots(figsize=(14, 8))
            poly = np.polyfit(years, df_kwara.yearly[feature], 1)
            fxn = np.poly1d(poly)
            ax.plot(years, fxn(years), color="r", label="Trend Line")
            ax.plot(years, df_kwara.yearly[feature], color="b", label="Actual Data")
            ax.set_title(f"Average {feature} from {years[0]}-{years[-1]} in KWARA")
            ax.set_xticks(years)
            ax.set_xticklabels(years, rotation=90)
            ax.set_ylabel(feature)
            ax.legend()
            st.pyplot(fig)

        st.markdown("---")

        # ANOVA Section
        st.header("ANOVA Analysis of Weather Data 📈")
        st.markdown(
            "### Comparing weather parameters across different towns in Kwara State (Confidence Level: 95%)"
        )

        # Get unique features
        weather_features = [
            "Relative Humidity  (%)",
            "Air Temperature (Max)",
            "Air Temperature (Min)",
            "Precipitation (mm)",
            "Solar Irradiance (MJ/m^2/day)",
            "Windspeed(m/s)MAX",
            "Windspeed(m/s)Min",
        ]

        for feature in weather_features:
            st.markdown(f"**ANOVA for {feature}**")
            alpha = 0.05
            frames = [w.yearly[feature] for w in [il, omu, okuta, patigi]]
            dat = pd.concat(frames, axis=1).drop("2024", errors="ignore")
            dat.columns = ["ILORIN", "OMU ARAN", "OKUTA", "PATIGI"]

            # Drop rows with NaN values to ensure f_oneway works correctly
            dat.dropna(inplace=True)

            F, p = f_oneway(dat["ILORIN"], dat["OMU ARAN"], dat["OKUTA"], dat["PATIGI"])

            st.write(f"F-statistic = {F:.4f}")
            st.write(f"p-value = {p}")
            if p < alpha:
                st.write(
                    "There is sufficient evidence that the four average measurements are not the same."
                )
            else:
                st.write("There is not enough evidence to reject the Null Hypothesis.")
            st.dataframe(dat.head())
            st.markdown("---")

        st.markdown("---")

        # Crop Yields Section
        st.header("Crop Yield Analysis 📊")
        with st.spinner("Loading and processing crop data..."):
            crop_data = pd.read_excel(crop_file, "Sheet1")
            yields_df = yield_df(crop_data)

        if yields_df is not None:
            st.success("Crop data loaded and processed successfully!")
            st.dataframe(yields_df)

            st.subheader("Total Yields of Each Crop (2011-2024)")
            fig, ax = plt.subplots(figsize=(12, 8))
            yields_df.sum().plot(
                kind="bar",
                ax=ax,
                title="Total Yields between 2011-2024",
                ylabel="Yields (Tons/Ha)",
            )
            st.pyplot(fig)

            st.markdown("---")

            st.header("Effect of Weather on Crop Yields")

            # Combine weather and crop data for relevant years
            combined_weather_crop = data.loc["2011":"2024"]

            crops_list = yields_df.columns.to_list()

            # if selected_crop:
            with st.form(key="my form"):
                selected_crop = st.selectbox("Select a crop to analyze:", crops_list)

                st.form_submit_button(label=f"Analyse")
                if selected_crop:
                    plot_crop(yields_df, combined_weather_crop, selected_crop)

except FileNotFoundError:
    st.error(
        f"The required Excel files ('{excel_file}' or '{crop_file}') were not found. Please ensure they are in the same directory as the script."
    )
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
