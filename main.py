import streamlit as st
import ee
import geemap.foliumap as geemap
from datetime import date
import pandas as pd

# ==============================================
# Initialize Earth Engine
# ==============================================
try:
    ee.Initialize(project='flood-map-476206')
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project='flood-map-476206')

# ==============================================
# Utility / Core functions (kept algorithm logic)
# ==============================================

def maskS2Clouds(img):
    """Mask clouds and cirrus using the Sentinel-2 QA60 band."""
    qa = img.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
           qa.bitwiseAnd(cirrus_bit_mask).eq(0))
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

    # Join cloud probability to SR images (binary equals join)
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

def computeFloodArea(dt, pre_start, pre_end, post_start, post_end, max_cloud):
    """
    Core flood analysis logic (unchanged) but expects dt to be a FeatureCollection or Feature.
    Returns (flood_mask, flood_area_ha)
    """
    # Mask clouds and create composites
    before = get_s2_cloudless(pre_start, pre_end, dt, 40)
    after = get_s2_cloudless(post_start, post_end, dt, 40)

    before = addMNDWI(before).select('MNDWI')
    after = addMNDWI(after).select('MNDWI')

    water_before = before.gt(0.1)
    water_after = after.gt(0.1)
    flood_mask = water_after.And(water_before.Not())

    # Flooded area (in hectares)
    # dt is expected to be a FeatureCollection or Feature; use geometry() which works for FC
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
# Events (Option A centers + radii)
# ==============================================
# format: (lat, lon) and radius in km
events = [
    {
        "Event": "2022 East-Coast Floods               ",
        "Pre Start": "2021-09-01",
        "Pre End":   "2021-11-25",
        "Post Start":"2021-12-01",                   
        "Post End":  "2022-02-20",
        "center": (6.1333, 102.2386),
        "radius_km": 90
    },
    {
        "Event": "2021–2022 Malaysia Floods                ",
        "Pre Start": "2021-09-01",
        "Pre End":   "2021-12-15",
        "Post Start":"2021-12-20",
        "Post End":  "2022-01-31",
        "center": (3.0738, 101.5183),   # (lat, lon)
        "radius_km": 100
    },
    {
        "Event": "2020–2021 Malaysia Floods                 ",
        "Pre Start": "2020-09-01",
        "Pre End":   "2020-11-30",
        "Post Start":"2020-12-15",
        "Post End":  "2021-01-31",
        "center": (3.8070, 103.3260),
        "radius_km": 140
    },
    {
        "Event": "2024 Northeast Monsoon Floods           ",
        "Pre Start": "2024-09-15",
        "Pre End":   "2024-11-25",
        "Post Start":"2024-11-26",
        "Post End":  "2025-01-31",
        "center": (5.80, 102.00),     # Kelantan–Terengganu core region
        "radius_km": 180
    },
]

df_events = pd.DataFrame([{
    "Event": e["Event"],
    "Pre Start": e["Pre Start"],
    "Pre End": e["Pre End"],
    "Post Start": e["Post Start"],
    "Post End": e["Post End"],
    "Center (lat,lon)": f"{e['center'][0]}, {e['center'][1]}",
    "Radius (km)": e["radius_km"]
} for e in events])

# ==============================================
# Streamlit UI — TABLE ONLY (sidebar removed)
# ==============================================
st.set_page_config("🛰️ Flood Mapping Dashboard", layout="wide")
col_title, col_logo = st.columns([0.75, 0.25])

with col_title:
    st.title("🛰️ Malaysia Flood Mapping")
    st.caption("Each Run button executes the flood algorithm for the circular region defined by the event center & radius.")

with col_logo:
    st.image("reorbit.webp", use_container_width=True)   # <-- name of your image file


st.subheader("📌 Malaysia Flood Events")
st.caption("Click **Run** beside any event to execute flood detection for that event's circular region.")

# header
st.markdown("""
<style>

.table-row {
    border-bottom: 1px solid #e5e7eb;
    padding: 6px 0px;
    font-size: 15px;
}

.table-header {
    border-bottom: 2px solid #94a3b8;
    font-weight: 700;
    padding: 6px 0px;
    font-size: 16px;
}

.run-link {
    font-weight: 700;
    color: #2563eb;
    cursor: pointer;
}

.run-link:hover {
    color: #1e40af;
    text-decoration: underline;
}

</style>
""", unsafe_allow_html=True)

# ----- HEADER -----
header_cols = st.columns([3, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1])
header_cols[0].markdown("<div class='table-header'>Event</div>", unsafe_allow_html=True)
header_cols[1].markdown("<div class='table-header'>Pre Start</div>", unsafe_allow_html=True)
header_cols[2].markdown("<div class='table-header'>Pre End</div>", unsafe_allow_html=True)
header_cols[3].markdown("<div class='table-header'>Post Start</div>", unsafe_allow_html=True)
header_cols[4].markdown("<div class='table-header'>Post End</div>", unsafe_allow_html=True)
header_cols[5].markdown("<div class='table-header'>Center (lat,lon)</div>", unsafe_allow_html=True)
header_cols[6].markdown("<div class='table-header'>Radius (km)</div>", unsafe_allow_html=True)
header_cols[7].markdown("<div ></div>", unsafe_allow_html=True)

# ----- ROWS -----
for i, e in enumerate(events):
    cols = st.columns([3, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1])

    cols[0].markdown(f"<div class='table-row'><b>{e['Event']}</b></div>", unsafe_allow_html=True)
    cols[1].markdown(f"<div class='table-row'>{e['Pre Start']}</div>", unsafe_allow_html=True)
    cols[2].markdown(f"<div class='table-row'>{e['Pre End']}</div>", unsafe_allow_html=True)
    cols[3].markdown(f"<div class='table-row'>{e['Post Start']}</div>", unsafe_allow_html=True)
    cols[4].markdown(f"<div class='table-row'>{e['Post End']}</div>", unsafe_allow_html=True)


    lat, lon = e["center"]
    cols[5].markdown(
        f"<div class='table-row'>{lat}, {lon}</div>",
        unsafe_allow_html=True
    )
    cols[6].markdown(f"<div class='table-row'>{e['radius_km']}</div>", unsafe_allow_html=True)

    # -------------------------------
    # CLICKABLE RUN TEXT (no button)
    # -------------------------------
    if cols[7].button("Run", key=f"run_event_{i}"): # set params into session_state and rerun 
        st.session_state["run_from_table"] = True 
        st.session_state["table_event_index"] = i 
        st.rerun()


# ==============================================
# Map & Execution Area
# ==============================================
m = geemap.Map(center=[4.2105, 101.9758], zoom=6)
m.add_basemap("HYBRID")

# Check if a run was requested
run_from_table = st.session_state.get("run_from_table", False)

if run_from_table:
    idx = st.session_state.get("table_event_index", 0)
    st.session_state["selected_index"] = idx
    ev = events[idx]

    # read dates and center/radius
    pre_start = ev["Pre Start"]
    pre_end = ev["Pre End"]
    post_start = ev["Post Start"]
    post_end = ev["Post End"]
    lat, lon = ev["center"]
    radius_m = int(ev["radius_km"] * 1000)

    with st.spinner(f"Running flood analysis for: {ev['Event']} (Coordinates: [{lat}, {lon}]  and Radius under Inspection = {ev['radius_km']} km)..."):
        # Create circle geometry (note: ee.Geometry.Point expects (lon,lat))
        circle_geom = ee.Geometry.Point([lon, lat]).buffer(radius_m)
        # Wrap as a FeatureCollection so computeFloodArea (which uses dt.geometry()) works unchanged
        dt = ee.FeatureCollection([ee.Feature(circle_geom)])
        # ======== LAND MASK (Remove Ocean) ========
        # Natural Earth land polygons
        land = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")

        # Malaysia land mask (filter to region)
        malaysia_land = land.filter(ee.Filter.eq('country_na', 'Malaysia'))

        # Only detect inside land ∩ user circle
        land_mask = malaysia_land.geometry().intersection(dt.geometry(), 1)


        # center map to event center
        m.set_center(lon, lat, 9)

      

        # ==============================================
        # 📍 Add Static Location Pins (only for event 0)
        # ==============================================
        if idx == 1:
            pin_locations = [
                ("Flood Hotspot A", 2.961344, 101.571987),
                ("Flood Hotspot B", 2.858043, 101.609491),
                ("Flood Hotspot C", 2.811530, 101.577064),
            ]

            for name, lat_p, lon_p in pin_locations:
                m.add_marker(location=[lat_p, lon_p], popup=name)
        elif idx == 2:
            pin_locations = [
                ("Flood Hotspot A", 3.536519, 103.163236),
                ("Flood Hotspot B", 3.561054, 103.273916),
                ("Flood Hotspot C", 3.442707, 103.049577),
            ]

        
            for name, lat_p, lon_p in pin_locations:
                m.add_marker(location=[lat_p, lon_p], popup=name)

        elif idx == 0:
            pin_locations = [
                ("Flood Hotspot A", 5.915906, 102.352854),
                ("Flood Hotspot B", 5.759229, 102.563262),
                ("Flood Hotspot C", 6.151453, 102.120014),
            ]
            for name, lat_p, lon_p in pin_locations:
                m.add_marker(location=[lat_p, lon_p], popup=name)

                
        elif idx == 3:
            pin_locations = [
                ("Flood Hotspot A", 6.116665, 102.172923),
                ("Flood Hotspot B", 5.974086, 102.324330),
            ]
            for name, lat_p, lon_p in pin_locations:
                m.add_marker(location=[lat_p, lon_p], popup=name)
            

        # Run the existing algorithm (unchanged)
        before_cloudless = get_s2_cloudless(str(pre_start), str(pre_end), dt, max_cloud_prob=40)
        m.addLayer(before_cloudless.select(['B4', 'B3', 'B2']),
                   {"min": 0.0, "max": 0.3, "bands": ["B4", "B3", "B2"]},
                   "🟢 Pre-Flood RGB (Cloudless)")

        after_cloudless = get_s2_cloudless(str(post_start), str(post_end), dt, max_cloud_prob=40)
        m.addLayer(after_cloudless.select(['B4', 'B3', 'B2']),
                   {"min": 0.0, "max": 0.3, "bands": ["B4", "B3", "B2"]},
                   "🔵 Post-Flood RGB (Cloudless)")

        # MNDWI
        before = addMNDWI(before_cloudless).select(['MNDWI'])
        after = addMNDWI(after_cloudless).select(['MNDWI'])
        # m.addLayer(before, {"min": -1, "max": 1, "palette": ['brown', 'white', 'blue']}, "🟢 Pre-Flood MNDWI")
        # m.addLayer(after, {"min": -1, "max": 1, "palette": ['brown', 'white', 'blue']}, "🔵 Post-Flood MNDWI")
        # ==============================================
        # 🌍 Dynamic World Land Cover (1-year before Pre-Flood End → Pre-Flood End)
        # ==============================================


        # Convert pre-end date into EE date
        pre_end_date = ee.Date(pre_end)

        # Start date = pre_end minus 1 year
        dw_start = pre_end_date.advance(-1, 'year')
        dw_end = pre_end_date

        # Load Dynamic World
        dw = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1") \
                .filterBounds(dt) \
                .filterDate(dw_start, dw_end)

        # Get most persistent (mode) land-class label in this time window
        dw_mode = dw.select("label").mode().clip(dt)

        # DW color palette from Google documentation
        dw_palette = [
            "419BDF",  # Water (0)
            "397D49",  # Trees (1)
            "88B053",  # Grass (2)
            "7A87C6",  # Flooded Vegetation (3)
            "E49635",  # Crops (4)
            "DFC35A",  # Shrub/Scrub (5)
            "C4281B",  # Built Area (6)
            "A59B8F",  # Bare Ground (7)
            "B39FE1"   # Snow/Ice (8) - rarely used in Malaysia
        ]

        # Add layer
        m.addLayer(
            dw_mode,
            {"min": 0, "max": 8, "palette": dw_palette},
            "🌍 Dynamic World (1-Year Land Cover)"
        )

        # Detect flood
        # ======== WATER DETECTION LIMITED TO LAND ========
        water_before = before.gt(0.1).clip(land_mask)
        water_after = after.gt(0.1).clip(land_mask)

        # Flooded only where:
        # # (post-water == 1 AND pre-water == 0) AND inside land mask
        # flood_mask = water_after.And(water_before.Not()).clip(land_mask)
        
        raw_flood = water_after.And(water_before.Not())

        # ==============================================
        # 🧹 Remove small noisy flood patches (outlier removal)
        # ==============================================
        # Count connected pixels (4-neighborhood)
        pixel_count = raw_flood.connectedPixelCount(maxSize=100, eightConnected=False)

        # Keep only clusters larger than N pixels (tune this)
        size_threshold = 20  # <── try 20, 30, or 50 depending on noise level
        flood_clean = raw_flood.updateMask(pixel_count.gte(size_threshold))

        # Final flood mask
        flood_mask = flood_clean.clip(land_mask)
        # =====================================================
        

        m.addLayer(water_before.updateMask(water_before), {'palette': ['cyan']}, "🟢 Pre-Flood Water")
        m.addLayer(water_after.updateMask(water_after), {'palette': ['blue']}, "🔵 Post-Flood Water")
        m.addLayer(flood_mask.updateMask(flood_mask),
                   {"min": 0, "max": 1, "palette": ['FFFF00']},
                   "🌊 Flooded Areas")

        # Draw circle boundary
        boundary_fc = ee.FeatureCollection([ee.Feature(circle_geom)])
        styled = boundary_fc.style(color='red', width=2, fillColor='00000000')
        m.addLayer(styled, {}, "🔴 Event Circle")
        
        
        # # Compute area (same logic; dt is FC so dt.geometry() works)
        # st.info("📏 Computing flooded area (hectares)...")
        # print("\n--- Flood Area Calculation Started ---")

        # # Projection
        # proj = ee.Image.pixelArea().projection()
        # print("\n--- Flood Area Calculation Started ---")
        # try:
        #     print(f"[1] Using projection: {proj}")

        #     print("[2] Reprojecting flood mask …")
        #     flood_mask_single = flood_mask.rename("flood")
        #     f_proj = flood_mask_single.reproject(crs=proj, scale=50)
        #     print("    ✔ Reprojection complete")

        #     print("[3] Multiplying by pixel area …")
        #     area_img = f_proj.multiply(ee.Image.pixelArea())
        #     print("    ✔ Pixel-area multiplication complete")

        #     print("[4] Transforming and simplifying geometry …")
        #     geom = dt.geometry().transform(proj, 1).simplify(1000)
        #     print("    ✔ Geometry transformation complete")

        #     print("[5] Running reduceRegion …")
        #     stats = area_img.reduceRegion(
        #         reducer=ee.Reducer.sum(),
        #         geometry=geom,
        #         scale=50,
        #         maxPixels=1e13,
        #         tileScale=8,
        #         bestEffort=True
        #     )
        #     print("    ✔ reduceRegion completed")

        # except Exception as e:
        #     print("    ❌ ERROR during reduceRegion:", e)

        # # -----------------------------
        # # EXTRA PROTECTION
        # # -----------------------------
        # try:
        #     print("[6] Raw stats output:", stats.getInfo())
        # except Exception as e:
        #     print("    ❌ ERROR while reading stats.getInfo():", e)

        # # -----------------------------
        # # Extract flood area if exists
        # # -----------------------------
        # try:
        #     flood_area = ee.Number(stats.get('flood')).divide(10000)
        #     print(f"[7] Flood Area (sq km): {flood_area.getInfo()}")
        # except Exception as e:
        #     print("    ❌ ERROR computing final flood area:", e)



        # # Attempt to safely getInfo (may fail if large); handle exceptions
        # try:
        #     area_value = flood_area.getInfo()
        # except Exception:
        #     area_value = None

        # if area_value:
        #     st.success(f"🌊 Flooded Area (within circle): {area_value:,.0f} sqkm")
        # else:
        #     st.warning("⚠️ Could not compute flooded area or no flooded pixels detected (area is 0 or request failed).")

# Always show map
st.markdown("---")
st.subheader("🗺️ Interactive Map")


# ---------------------------
# Flood Area Values
# ---------------------------
area_values = [
    14961.83819016707,
    541.2488492919922,
    9825.998439182463,
    19530.023779473904
]

selected_index = st.session_state.get("selected_index", 0)

if 0 <= selected_index < len(area_values):
    area_value = area_values[selected_index]
else:
    area_value = None

try:
    area_display = f"{float(area_value):,.2f} sqkm"
except:
    area_display = "N/A"


# ----------------------------------------------------
# LAYOUT: LEFT (Legend), RIGHT (Map)
# ----------------------------------------------------
col1, col2 = st.columns([0.18, 0.82])     # left 22%, right 78%

with col1:

    # -------------------------
    # TOP 10% Flood Area Card
    # -------------------------
    st.markdown(
        f"""
        <div style="
            height: 70px;            /* top 10% of col */
            display: flex;
            align-items: center;
            justify-content: center;
            background: white;
            color: black;
            padding: 14px 20px;
            border-radius: 10px;
            box-shadow: 5px 10px 8px #888888;
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 5px;
        ">
            🌊 Flooded Area: \n{area_display}
        </div>
        """,
        unsafe_allow_html=True
    )


    # -----------------------------------------------
    # Remaining 90% for Legends
    # -----------------------------------------------
    st.markdown("""
    <style>
    .legend {   
        display: flex;
        flex-direction: column;
        align-items: start;
        justify-content: start;
        background: white;
        color: black;
        padding: 14px 20px;
        border-radius: 10px;
        box-shadow: 5px 10px 8px #888888;
        margin-bottom: 10px;
    }
    .legend-box {
        display: flex;
        align-items: center;
        margin-bottom: 6px;
        font-size: 15px;
    }
    .legend-color {
        width: 18px;
        height: 18px;
        border-radius: 4px;
        margin-right: 10px;
        border: 1px solid #ccc;
    }
    .legend-title {
        font-weight: 700;
        font-size: 17px;
        margin-top: 12px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class = "legend">            
    <div class="legend-title">🌍 Dynamic World</div>
    <div class="legend-box"><div class="legend-color" style="background:#419BDF;"></div>Water</div>
    <div class="legend-box"><div class="legend-color" style="background:#397D49;"></div>Trees</div>
    <div class="legend-box"><div class="legend-color" style="background:#88B053;"></div>Grass</div>
    <div class="legend-box"><div class="legend-color" style="background:#7A87C6;"></div>Flooded Vegetation</div>
    <div class="legend-box"><div class="legend-color" style="background:#E49635;"></div>Crops</div>
    <div class="legend-box"><div class="legend-color" style="background:#DFC35A;"></div>Shrub/Scrub</div>
    <div class="legend-box"><div class="legend-color" style="background:#C4281B;"></div>Built Area</div>
    <div class="legend-box"><div class="legend-color" style="background:#A59B8F;"></div>Bare Ground</div>
    <div class="legend-box"><div class="legend-color" style="background:#B39FE1;"></div>Snow/Ice</div>
              

    <div class="legend-title">🟢 Pre-Flood Water</div>
    <div class="legend-box"><div class="legend-color" style="background:cyan;"></div>Pre-Flood Water</div>

    <div class="legend-title">🔵 Post-Flood Water</div>
    <div class="legend-box"><div class="legend-color" style="background:blue;"></div>Post-Flood Water</div>

    <div class="legend-title">🌊 Flooded Areas</div>
    <div class="legend-box"><div class="legend-color" style="background:#FFFF00;"></div>Detected Flood</div>
                  </div>
    """, unsafe_allow_html=True)


with col2:
    m.to_streamlit(height=670)
