import streamlit as st
import main
from pathlib import Path
from PIL import Image

st.set_page_config(
    page_title = "Border Surveillance System",
    layout     = "wide",
)

st.markdown("",
    unsafe_allow_html=True,
)

ALGO_OPTIONS = {
    "MAD — Mean Absolute Difference": (
        "mad",
        "Fast pixel-level baseline across R/G/B channels.",
    ),
    "Log-Ratio (SAR)": (
        "ratio",
        "Logarithmic ratio — optimal for Sentinel-1 SAR imagery.",
    ),
    "NDVI — Vegetation Change": (
        "ndvi",
        "Vegetation-index change proxy for forest/agricultural zones.",
    ),
    "K-Means Clustering": (
        "kmeans",
        "Unsupervised ML — groups pixels into 5 change-intensity clusters.",
    ),
    "SVM — Support Vector Machine": (
        "svm",
        "RBF-kernel SVM on 13-D per-pixel features with self-supervised labels.",
    ),
    "Random Forest": (
        "random_forest",
        "80-tree ensemble on 22-D features; outputs P(change) per pixel.",
    ),
    "Siamese Network": (
        "siamese",
        "Shared encoder → L2 distance in 18-D feature space.",
    ),
    "AlphaEarth Foundation Model": (
        "alphaearth",
        "ViT patch embedding → cosine distance (D=64, P=16).",
    ),
}

# Header 
st.markdown('<p class="main-title"> Border Surveillance System</p>', unsafe_allow_html=True)

# Sidebar 
st.sidebar.header(" Configuration")

# Preset selector
preset_options = {
    "Custom coordinates": None,
    "Jammu LoC":          (32.72,  74.86),
    "Kabul Border":       (34.52,  69.18),
    "Pakistan LoC":       (33.68,  73.05),
    "Bhutan-India":       (27.47,  89.64),
    "Bangladesh Border":  (23.73,  90.39),
    "Rann of Kutch":      (24.88,  67.03),
}
preset_choice = st.sidebar.selectbox(" Border Region Preset", list(preset_options.keys()))

if preset_options[preset_choice] is not None:
    default_lat, default_lon = preset_options[preset_choice]
else:
    default_lat, default_lon = 32.72, 74.86

# Coordinates 
st.sidebar.markdown("**Target Coordinates**")
lat = st.sidebar.number_input("Latitude  (°N)",  value=default_lat, format="%.4f",
                               min_value=-90.0,  max_value=90.0)
lon = st.sidebar.number_input("Longitude (°E)", value=default_lon, format="%.4f",
                               min_value=-180.0, max_value=180.0)

# Satellite
st.sidebar.markdown("**Satellite**")
sat_display = st.sidebar.selectbox(
    "Satellite Type",
    ["S2 — Sentinel-2 (Optical)", "S1 — Sentinel-1 (SAR)"],
)
sat_type = "S2" if sat_display.startswith("S2") else "S1"

# Algorithm dropdown 
st.sidebar.markdown("**Select Algorithm**")
algo_display = st.sidebar.selectbox(
    "Algorithm",
    list(ALGO_OPTIONS.keys()),
    index=0,
)
algo_key, algo_desc = ALGO_OPTIONS[algo_display]

# Threshold
st.sidebar.markdown("**Sensitivity**")
threshold = st.sidebar.slider(
    "Change Threshold (0–255)",
    min_value = 5,
    max_value = 100,
    value     = 40,
)

#  Run button 
run_clicked = st.sidebar.button("▶  Run Surveillance Scan", use_container_width=True)

st.markdown(
    f'<div class="algo-desc">'
    f'<strong style="color:#00c88c">{algo_display}</strong><br>{algo_desc}'
    f'</div>',
    unsafe_allow_html=True,
)

if run_clicked:
    output_file = "ui_report.png"

    with st.spinner(f"Running {algo_display} on {lat:.4f}°N, {lon:.4f}°E …"):
        try:
            main.run_system(
                lat       = lat,
                lon       = lon,
                preset    = None,
                satellite = sat_type,
                algo      = algo_key,
                threshold = float(threshold),
                seed      = 42,
                output    = output_file,
                show      = False,
            )

            if Path(output_file).exists():
                report_img = Image.open(output_file)
                st.image(
                    report_img,
                    caption = (
                        f"{algo_display} . "
                        f"{'Sentinel-2' if sat_type == 'S2' else 'Sentinel-1'} . "
                        f"{lat:.4f}°N, {lon:.4f}°E"
                    ),
                    use_container_width = True,
                )

                with open(output_file, "rb") as fh:
                    st.download_button(
                        label     = "⬇️  Download Report PNG",
                        data      = fh,
                        file_name = f"bss_{algo_key}_{lat:.2f}_{lon:.2f}.png",
                        mime      = "image/png",
                    )
            else:
                st.error("Report file was not created. Check the terminal for errors.")

        except ImportError as e:
            st.error(
                f"Missing dependency for **{algo_display}**: `{e}`\n\n"
                "Run `pip install scikit-learn scipy` and restart the app."
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.exception(e)

else:
    st.info(
        " Configure the target coordinates and algorithm in the sidebar, "
        "then click **Run Surveillance Scan**.",
    )