import streamlit as st
import ee
import geemap.foliumap as geemap
from datetime import date

# ==============================================
# Initialize Earth Engine
# ==============================================
try:
    ee.Initialize(project='flood-map-476206')
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project='flood-map-476206')

# ==============================================
# Helper Functions
# ==============================================

import pandas as pd

# ===============================
# Flood Event Table
# ===============================
events = [
    {
        "Event": "2021–2022 Malaysia Floods",
        "Pre Start": "2021-11-01",
        "Pre End": "2021-12-15",
        "Post Start": "2021-12-20",
        "Post End": "2022-01-31",
    },
    {
        "Event": "2020–2021 Malaysia Floods",
        "Pre Start": "2020-10-01",
        "Pre End": "2020-11-30",
        "Post Start": "2020-12-15",
        "Post End": "2021-01-31",
    },
    {
        "Event": "2022 East-Coast Floods",
        "Pre Start": "2022-02-01",
        "Pre End": "2022-02-20",
        "Post Start": "2022-02-26",
        "Post End": "2022-03-15",
    },
    {
        "Event": "2015 East Malaysia Floods",
        "Pre Start": "2014-12-01",
        "Pre End": "2015-01-16",
        "Post Start": "2015-02-07",
        "Post End": "2015-02-28",
    },
]

df_events = pd.DataFrame(events)

st.subheader("📌 Malaysia Flood Events (Quick Run)")
st.caption("Click **Run** beside any event to automatically execute flood detection for that window.")

# --- HEADER ROW ---
header_cols = st.columns([3, 2, 2, 2, 2, 1])
header_cols[0].markdown("**Event**")
header_cols[1].markdown("**Pre Start**")
header_cols[2].markdown("**Pre End**")
header_cols[3].markdown("**Post Start**")
header_cols[4].markdown("**Post End**")
header_cols[5].markdown("")

# --- EVENT ROWS ---
for i, row in df_events.iterrows():
    cols = st.columns([3, 2, 2, 2, 2, 1])
    cols[0].markdown(f"**{row['Event']}**")
    cols[1].write(row["Pre Start"])
    cols[2].write(row["Pre End"])
    cols[3].write(row["Post Start"])
    cols[4].write(row["Post End"])

    if cols[5].button("Run", key=f"run_event_{i}"):
        # override sidebar dates
        pre_start = row["Pre Start"]
        pre_end = row["Pre End"]
        post_start = row["Post Start"]
        post_end = row["Post End"]

        st.session_state["run_from_table"] = True
        st.session_state["table_pre_start"] = pre_start
        st.session_state["table_pre_end"] = pre_end
        st.session_state["table_post_start"] = post_start
        st.session_state["table_post_end"] = post_end

        st.rerun()
        
def maskS2Clouds(img):
    """Mask clouds and cirrus using the Sentinel-2 QA60 band."""
    # QA60 band contains cloud and cirrus bits:
    # Bit 10 → clouds, Bit 11 → cirrus
    qa = img.select('QA60')
    
    # Create bit masks for clouds and cirrus
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    
    # Both flags should be set to 0, indicating clear conditions
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
           qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    
    # Apply mask and scale reflectance to 0–1
    return img.updateMask(mask).divide(10000)

def addMNDWI(img):
    """Compute MNDWI band."""
    return img.addBands(img.normalizedDifference(['B3', 'B11']).rename('MNDWI'))

def get_s2_cloudless(start, end, geometry, max_cloud_prob=40):
    s2_sr = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(geometry) \
        .filterDate(start, end)

    s2_clouds = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY') \
        .filterBounds(geometry) \
        .filterDate(start, end)

    # Join cloud probability to SR images
    joined = ee.Join.saveFirst('cloud_mask').apply(
        primary=s2_sr,
        secondary=s2_clouds,
        condition=ee.Filter.equals(
            leftField='system:index',
            rightField='system:index'
        )
    )

    def mask_clouds(img):
        cloud_mask = ee.Image(img.get('cloud_mask')).select('probability')
        mask = cloud_mask.lt(max_cloud_prob)
        return img.updateMask(mask).divide(10000)

    clean = ee.ImageCollection(joined).map(mask_clouds)
    return clean.median().clip(geometry)


def computeFloodArea(dt, pre_start, pre_end, post_start, post_end, max_cloud, state, district):
    """Core flood analysis logic replicated from your Earth Engine script."""
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(dt) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud))
    
    # Mask clouds
    before = get_s2_cloudless(pre_start, pre_end, dt, 40)
    after = get_s2_cloudless(post_start, post_end, dt, 40)
    
    before = addMNDWI(before).select('MNDWI')
    after = addMNDWI(after).select('MNDWI')

    water_before = before.gt(0.1)
    water_after = after.gt(0.1)
    flood_mask = water_after.And(water_before.Not())

    # Flooded area (in hectares)
    stats = flood_mask.rename('flood').multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=dt.geometry().simplify(1000),
        scale=30,
        maxPixels=1e13,
        tileScale=16,
        bestEffort=True
    )
    flood_area = ee.Number(stats.get('flood')).divide(10000)
    return flood_mask, flood_area

# ==============================================
# Load Malaysia Boundaries
# ==============================================
malaysia = ee.FeatureCollection('FAO/GAUL/2015/level0') \
    .filter(ee.Filter.eq('ADM0_NAME', 'Malaysia')) \
    .geometry()

# ==============================================
# Streamlit UI
# ==============================================

st.set_page_config("🛰️ Flood Mapping Dashboard", layout="wide")
st.title("🛰️ Flood Mapping Dashboard")
st.caption("Using Sentinel-2 imagery & MNDWI for optical flood detection")

with st.sidebar:
    st.image("./reorbit.webp", use_container_width=True)

    st.header("🌏 Malaysia Flood Map")

    
    st.header("📅 Date Range")
    pre_col1, pre_col2 = st.columns(2)
    pre_start = pre_col1.date_input("Pre-Flood Start", date(2022, 3, 25))
    pre_end = pre_col2.date_input("Pre-Flood End", date(2022, 5, 15))
    
    post_col1, post_col2 = st.columns(2)
    post_start = post_col1.date_input("Post-Flood Start", date(2022, 5, 16))
    post_end = post_col2.date_input("Post-Flood End", date(2022, 7, 22))
    
    st.header("☁️ Cloud Filter")
    max_cloud = st.slider("Max Cloud %", 0, 100, 50, 5)
    
    run_btn = st.button("🚀 Run Flood Analysis")

# ==============================================
# Map and Results Section
# ==============================================
m = geemap.Map(center=[4.2105, 101.9758], zoom=6)
m.add_basemap("HYBRID")

run_from_sidebar = run_btn
run_from_table = st.session_state.get("run_from_table", False)

if run_from_sidebar or run_from_table:

    if run_from_table:
        pre_start = st.session_state["table_pre_start"]
        pre_end = st.session_state["table_pre_end"]
        post_start = st.session_state["table_post_start"]
        post_end = st.session_state["table_post_end"]

    with st.spinner("Running flood analysis..."):
        dt = malaysia
        district_geom = malaysia.bounds()

        centroid = district_geom.centroid(1000).coordinates().getInfo()
        m.set_center(110.0, 2.5, 6) 

        # Sentinel-2 collection
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(dt) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud))

        # Step 1: Pre-flood composite
        st.info("🟢 Generating Pre-Flood composite...")
        before_cloudless = get_s2_cloudless(str(pre_start), str(pre_end), dt, max_cloud_prob=40)
        m.addLayer(before_cloudless.select(['B4', 'B3', 'B2']),
                {"min": 0.0, "max": 0.3, "bands": ["B4", "B3", "B2"]},
                "🟢 Pre-Flood RGB (Cloudless)")

        after_cloudless = get_s2_cloudless(str(post_start), str(post_end), dt, max_cloud_prob=40)
        m.addLayer(after_cloudless.select(['B4', 'B3', 'B2']),
                {"min": 0.0, "max": 0.3, "bands": ["B4", "B3", "B2"]},
                "🔵 Post-Flood RGB (Cloudless)")

        # Step 3: Add MNDWI layers
        st.info("🌊 Calculating MNDWI layers...")
        before = addMNDWI(before_cloudless).select(['B4', 'B3', 'B2', 'MNDWI'])
        after = addMNDWI(after_cloudless).select(['B4', 'B3', 'B2', 'MNDWI'])
        m.addLayer(before.select('MNDWI'),
                   {"min": -1, "max": 1, "palette": ['brown', 'white', 'blue']},
                   "🟢 Pre-Flood MNDWI")
        m.addLayer(after.select('MNDWI'),
                   {"min": -1, "max": 1, "palette": ['brown', 'white', 'blue']},
                   "🔵 Post-Flood MNDWI")

        # Step 4: Detect flood areas
        st.info("💧 Detecting flooded regions...")
        water_before = before.select('MNDWI').gt(0.1)
        water_after = after.select('MNDWI').gt(0.1)
        flood_mask = water_after.And(water_before.Not())

        # Add water masks
        m.addLayer(water_before.updateMask(water_before),
                   {'palette': ['cyan']}, "🟢 Pre-Flood Water")
        m.addLayer(water_after.updateMask(water_after),
                   {'palette': ['blue']}, "🔵 Post-Flood Water")

        # Add flood mask
        m.addLayer(flood_mask.updateMask(flood_mask),
               {"min": 0, "max": 1, "palette": ['000000', 'FFFF00']},
               "🌊 Flooded Areas")

        # Add district boundary
        boundary_fc = ee.FeatureCollection([ee.Feature(dt)])
        styled = boundary_fc.style(color='red', width=2, fillColor='00000000')

        m.addLayer(styled, {}, "🗺️ Boundary")

        # Step 5: Compute flooded area
        # st.info("📏 Computing flood area...")
        # stats = flood_mask.rename('flood').multiply(ee.Image.pixelArea()).reduceRegion(
        #     reducer=ee.Reducer.sum(),
        #     geometry = ee.Feature(dt).geometry().simplify(5000),
        #     scale=30,
        #     maxPixels=1e13,
        #     tileScale=16,
        #     bestEffort=True
        # )
        # flood_area = ee.Number(stats.get('flood')).divide(10000)
        # area_value = flood_area.getInfo()

        # # Show results
        # if area_value:
        #     st.success(f"🌊 **Flooded Area: {area_value:,.0f} ha**")
        # else:
        #     st.warning("⚠️ Could not compute flooded area.")


st.markdown("---")
st.subheader("🗺️ Interactive Map")
m.to_streamlit(height=650)
