import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats



# Page setup


st.set_page_config(page_title="Histogram Distribution Fitter", layout="wide")
st.title("Fit a Distribution to your Histogram!")
st.write("""
To get started: Either enter your data manually or upload a CSV file.
Select your preferred distribution to fit the data, then view your histogram with the fitted curve.
You can also manually adjust parameters using the sliders.
""")


# Sidebar input

st.sidebar.header("Data Input")
input_mode = st.sidebar.radio("Input method", ["Manual Entry", "Upload CSV"])

data = np.array([])

if input_mode == "Manual Entry":
    data_text = st.sidebar.text_area(
        "Enter numbers separated by commas, spaces, or newlines"
    )
    if data_text:
        try:
            data = np.array([float(x) for x in data_text.replace(',', ' ').split()])
        except:
            st.sidebar.error(
                "Invalid data. Please enter numbers separated by commas, spaces, or newlines."
            )

elif input_mode == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV (first column used)", type=["csv", "txt"]
    )
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, header=None)
            data = df.values.flatten()
            data = data[np.isfinite(data)]
        except:
            st.sidebar.error(
                "Unable to read file. Make sure it's a CSV with numeric data."
            )




# Histogram controls


bins = st.sidebar.slider("Histogram bars", min_value=5, max_value=100, value=30)



# Distribution selection


st.sidebar.header("Distribution Fit")
DIST_MAPPING = {
"Normal": "norm",
"Exponential": "expon",
"Gamma": "gamma",
"Weibull Min": "weibull_min",
"Weibull Max": "weibull_max",
"Log Normal": "lognorm",
"Beta": "beta",
"Pareto": "pareto",
"Uniform": "uniform",
"Chi-Square": "chi2",
"Double Exponential (Laplace)": "laplace",
"Rayleigh": "rayleigh",
}
dist_display_name = st.sidebar.selectbox("Distribution to fit", list(DIST_MAPPING.keys()))
dist_name = DIST_MAPPING[dist_display_name]

run_fit = st.sidebar.button("Run Fit")

if data.size == 0:
    st.warning("No data available. Enter numbers or upload a CSV.")
else:
    st.write(f"Data size: {len(data)}")
    st.write("Preview (first 200 values):")
    st.write(data[:200] if len(data) > 200 else data)


# Histogram plot setup

fig, ax = plt.subplots(figsize=(8, 4))
counts, bin_edges, _ = ax.hist(data, bins=bins, density=True, alpha=0.5, edgecolor='k', label="Data histogram")
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

fitted_params = None
average_error = None

if run_fit:
    # get distribution object
    dist_obj = getattr(stats, dist_name)
    try:
        params = dist_obj.fit(data)
        fitted_params = params

        # handle shape, loc, scale parameters
        if len(params) >= 2:
            shape_args = params[:-2]
            loc = params[-2]
            scale = params[-1]
        else:
            shape_args = []
            loc = 0
            scale = params[-1] if len(params) == 1 else 1

        # PDF for plotting
        x = np.linspace(bin_edges[0], bin_edges[-1], 400)
        pdf = dist_obj.pdf(x, *shape_args, loc=loc, scale=scale)
        pdf_centers = dist_obj.pdf(bin_centers, *shape_args, loc=loc, scale=scale)

        ax.plot(x, pdf, color='crimson', lw=2, label=f"{dist_display_name} fit")

        # Average Error: mean signed difference between histogram density and PDF at bin centers
        average_error = np.mean(counts - pdf_centers)

    except Exception as e:
        st.error(f"Fit failed: {e}")

ax.set_title("Histogram with Fitted Curve")
ax.legend()
st.pyplot(fig)


# Paramteres and error
if fitted_params is not None:
    st.subheader("Fitted Parameters")
    st.write("Distribution:", dist_display_name)
    st.write("Parameters (shape(s)..., loc, scale):")
    st.write(np.array(fitted_params, dtype=object))
    if average_error is not None:
        st.write(f"Average Error between histogram and PDF: {average_error:.6f}")

  
    # Manual sliders for fitting

    st.subheader("Manual Fit Sliders")
    manual_params = []
    for i, p in enumerate(fitted_params):
        min_val = float(p * 0.5)
        max_val = float(p * 1.5) if p != 0 else 1.0
        manual_params.append(st.slider(f"Parameter {i+1}", min_val, max_val, float(p)))

    manual_params = tuple(manual_params)

    # Plot manual fit
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.hist(data, bins=bins, density=True, alpha=0.5, edgecolor='k', label="Data histogram")

    # Recompute shape, loc, scale from sliders
    if len(manual_params) >= 2:
        shape_args = manual_params[:-2]
        loc = manual_params[-2]
        scale = manual_params[-1]
    else:
        shape_args = []
        loc = 0
        scale = manual_params[-1] if len(manual_params) == 1 else 1

    pdf_manual = dist_obj.pdf(bin_centers, *shape_args, loc=loc, scale=scale)
    x = np.linspace(bin_edges[0], bin_edges[-1], 400)
    pdf_manual_plot = dist_obj.pdf(x, *shape_args, loc=loc, scale=scale)
    ax2.plot(x, pdf_manual_plot, color='green', lw=2, linestyle='--', label="Manual Fit")
    ax2.set_title("Histogram with Manual Fit")
    ax2.legend()
    st.pyplot(fig2)

    manual_avg_error = np.mean(counts - pdf_manual)
    st.write(f"Average Error (manual fit): {manual_avg_error:.6f}")

