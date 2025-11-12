import streamlit as st
import ee
import geemap.foliumap as geemap
from datetime import date

# ==============================================
# Initialize Earth Engine
# ==============================================
try:
    ee.Initialize(project='ee-bhupendrarulekiller14')
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project='ee-bhupendrarulekiller14')

# ==============================================
# Helper Functions
# ==============================================

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

def computeFloodArea(dt, pre_start, pre_end, post_start, post_end, max_cloud, state, district):
    """Core flood analysis logic replicated from your Earth Engine script."""
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(dt) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud))
    
    # Mask clouds
    before = s2.filterDate(pre_start, pre_end).map(maskS2Clouds).mosaic().clip(dt)
    after = s2.filterDate(post_start, post_end).map(maskS2Clouds).mosaic().clip(dt)
    
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
# Load India District Data
# ==============================================
ind_dt = ee.FeatureCollection("users/suryadeepsingh/malaysia_singapore_brunei_Districts_level_2")

# ==============================================
# Streamlit UI
# ==============================================

st.set_page_config("🛰️ Flood Mapping Dashboard", layout="wide")
st.title("🛰️ Flood Mapping Dashboard")
st.caption("Using Sentinel-2 imagery & MNDWI for optical flood detection")

with st.sidebar:
    st.header("🌍 Select Region")
    # Now shape0 = country group (e.g., malaysia_singapore_brunei)
    # shape1 = first-level admin (e.g., Belait)
    # shape2 = sub-district (e.g., Kuala Belait)
    # shapegroup = country code (e.g., BRN)

    countries = sorted(list(set(ind_dt.aggregate_array('shapegroup').getInfo())))
    country = st.selectbox("Country (shapegroup)", countries, index=0)

    if country:
        regions = sorted(ind_dt.filter(ee.Filter.eq('shapegroup', country))
                            .aggregate_array('shape1').getInfo())
        region = st.selectbox("Region (shape1)", regions, index=0)
        
        subregions = sorted(ind_dt.filter(ee.Filter.eq('shape1', region))
                            .aggregate_array('shape2').getInfo())
        subregion = st.selectbox("Subregion (shape2)", subregions, index=0)
    else:
        region = subregion = None

    
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
m = geemap.Map(center=[21, 78], zoom=5)
m.add_basemap("HYBRID")

if run_btn and region and subregion:
    with st.spinner("Running flood analysis..."):
        dt = ind_dt.filter(ee.Filter.And(
            ee.Filter.eq('shapegroup', country),
            ee.Filter.eq('shape1', region),
            ee.Filter.eq('shape2', subregion)
        ))


        # Center and zoom to the district
        district_geom = dt.geometry().bounds()
        centroid = district_geom.centroid(1000).coordinates().getInfo()
        m.set_center(centroid[0], centroid[1], 9)

        # Sentinel-2 collection
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(dt) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud))

        # Step 1: Pre-flood composite
        st.info("🟢 Generating Pre-Flood composite...")
        before = s2.filterDate(str(pre_start), str(pre_end)).map(maskS2Clouds).median().clip(dt)
        m.addLayer(before.select(['B4', 'B3', 'B2']),
                   {"min": 0.0, "max": 0.3, "bands": ["B4", "B3", "B2"]},
                   "🟢 Pre-Flood RGB")

        # Step 2: Post-flood composite
        st.info("🔵 Generating Post-Flood composite...")
        after = s2.filterDate(str(post_start), str(post_end)).map(maskS2Clouds).median().clip(dt)
        m.addLayer(after.select(['B4', 'B3', 'B2']),
                   {"min": 0.0, "max": 0.3, "bands": ["B4", "B3", "B2"]},
                   "🔵 Post-Flood RGB")

        # Step 3: Add MNDWI layers
        st.info("🌊 Calculating MNDWI layers...")
        before = addMNDWI(before).select(['B4', 'B3', 'B2', 'MNDWI'])
        after = addMNDWI(after).select(['B4', 'B3', 'B2', 'MNDWI'])
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
        m.addLayer(dt.style(color='red', fillColor='00000000'), {}, f"🗺️ {subregion} Boundary")

        # Step 5: Compute flooded area
        st.info("📏 Computing flood area...")
        stats = flood_mask.rename('flood').multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=dt.geometry().simplify(1000),
            scale=30,
            maxPixels=1e13,
            tileScale=16,
            bestEffort=True
        )
        flood_area = ee.Number(stats.get('flood')).divide(10000)
        area_value = flood_area.getInfo()

        # Show results
        if area_value:
            st.success(f"🌊 **Flooded Area in {subregion}, {region}: {area_value:,.0f} ha**")
        else:
            st.warning("⚠️ Could not compute flooded area.")


st.markdown("---")
st.subheader("🗺️ Interactive Map")
m.to_streamlit(height=650)
