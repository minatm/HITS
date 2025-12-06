import streamlit as st
import pandas as pd
import numpy as np
import folium
import pgeocode
import time
from geopy.geocoders import Nominatim
from streamlit_folium import st_folium
import gurobipy as gp
from gurobipy import GRB
import random
from math import sqrt
import matplotlib.pyplot as plt
import datetime
import calendar
import re 
import matplotlib.colors as mcolors

# Set a custom color scheme for consistency
COLOR_HI_HP = 'darkred'
COLOR_HI_LP = 'darkorange'
COLOR_LI_HP = 'darkblue'
COLOR_LI_LP = 'darkgreen'
COLOR_DEPOT = 'darkgreen'
COLOR_ROUTE = 'black'
# Consistent color for all schools in the daily iterative map
COLOR_SCHOOL_UNIFORM = 'darkblue'

# Set Matplotlib style for a cleaner look
plt.style.use('ggplot')


# ---------------------------------------------------------
#        PLANNING CONSTANTS
# ---------------------------------------------------------
PLANNING_YEAR = 2026
# Anchor date is the start of the planning horizon (Week 1 in the model)
ANCHOR_DATE = datetime.date(PLANNING_YEAR, 1, 1)

# --- CSV COLUMN INDICES (1-based input, 0-based code index) ---
SCHOOL_NAME_COL_INDEX = 2  # 3rd column
PRIORITY_COL_INDEX = 9     # 10th column
REFERRAL_DATE_COL_INDEX = 10 # 11th column

# ---------------------------------------------------------
#        HELPER FUNCTIONS
# ---------------------------------------------------------

def clean_school_name(name):
    """
    Removes the '(X) ' prefix and converts the school name to uppercase 
    for case-insensitive matching with the JSON data.
    """
    if isinstance(name, str):
        # 1. Remove '(number) ' prefix
        cleaned_name = re.sub(r'^\(\d+\)\s*', '', name).strip()
        # 2. Convert to uppercase for case-insensitive match
        return cleaned_name.upper()
    # If not a string (e.g., None, NaN), return an empty string for safety
    return ""

def calculate_r_value(referral_date_str, anchor_date):
    """Calculates r[i] as the number of weeks between the referral date and the anchor date (Jan 1st)."""
    if pd.isna(referral_date_str) or str(referral_date_str).strip() == "":
        return None
    try:
        # Referral date is expected as YYYY-MM-DD
        referral_date = datetime.datetime.strptime(str(referral_date_str), '%Y-%m-%d').date()
    except ValueError:
        return None # Drop invalid dates

    # r[i] = (Referral Date - ANCHOR_DATE) in weeks.
    days_to_anchor = (referral_date - anchor_date).days
    r_val = days_to_anchor / 7.0
    
    return r_val


# ---------------------------------------------------------
#        PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="HITS: Advanced Clinician Scheduling Tool", layout="wide")

# --- Initialize Session State ---
if 'optimization_ran' not in st.session_state:
    st.session_state.optimization_ran = False
if 'optimization_success' not in st.session_state:
    st.session_state.optimization_success = False
if 'final_daily' not in st.session_state:
    st.session_state.final_daily = None
if 'final_daily_var' not in st.session_state:
    st.session_state.final_daily_var = None
if 'final_daily_z_var' not in st.session_state:
    st.session_state.final_daily_z_var = None
if 'x_li_weekly' not in st.session_state:
    st.session_state.x_li_weekly = {}
if 'aux1_weekly' not in st.session_state:
    st.session_state.aux1_weekly = {}
if 'tr_weekly' not in st.session_state:
    st.session_state.tr_weekly = {}

# --- NEW: Context Snapshots for Visualization Consistency ---
if 'saved_student_school' not in st.session_state:
    st.session_state.saved_student_school = {}
if 'saved_school_names' not in st.session_state:
    st.session_state.saved_school_names = {}
if 'saved_students' not in st.session_state:
    st.session_state.saved_students = []
if 'saved_theta' not in st.session_state:
    st.session_state.saved_theta = {}
if 'saved_p' not in st.session_state:
    st.session_state.saved_p = {}
if 'saved_I' not in st.session_state:
    st.session_state.saved_I = {}
if 'saved_assigned_schools_df' not in st.session_state:
    st.session_state.saved_assigned_schools_df = None
    
# NEW: Session state for editable student counts and tracking clinician change
if 'editable_counts' not in st.session_state:
    st.session_state.editable_counts = None
if 'last_clinician_id' not in st.session_state:
    st.session_state.last_clinician_id = None

# --- Callback to reset optimization on Clinician ID change ---
def reset_optimization_state():
    """Resets the state flags that control output visualization."""
    st.session_state.optimization_ran = False
    st.session_state.optimization_success = False
# -----------------------------------------------------------------


st.markdown(
    """
    <style>
        .small-font { font-size:14px !important; }
        .stNumberInput > div > input { font-size:14px !important; }
        .stTextInput > div > input { font-size:14px !important; }
        .stMarkdown { font-size:14px !important; }
        .stDataFrame { font-size:14px !important; }
        .reportview-container .main .block-container{
            padding-top: 1rem;
            padding-right: 1rem;
            padding-left: 1rem;
            padding-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌟 HITS: Clinician Scheduling Tool 🗓️")
st.markdown(
    """
This dashboard visualizes the **optimal schedules** for clinicians enrolled in the HITS Program**.
"""
)

# ---------------------------------------------------------
#        SIDEBAR — FILE UPLOADS
# ---------------------------------------------------------
st.sidebar.header("📂 Data File Uploads", divider="gray")

slp_file = st.sidebar.file_uploader("1. Upload Clinician data file (slp.json)", type="json")
school_file = st.sidebar.file_uploader("2. Upload School data file (schools.json)", type="json")
student_data_file = st.sidebar.file_uploader("3. Upload Student data file (CSV)", type="csv")

if not slp_file or not school_file or not student_data_file:
    st.warning("🚨 Please upload all required files (slp.json, schools.json, and student data CSV) to enable the tool. Note that slp.json includes the clinician availibility, schools.json includes the school name, address, and the assigned clinician, and student data csv includes the current waitlist of each school, along with the referral dates.")
    st.stop()

# --- Data Loading ---
try:
    slp_df = pd.read_json(slp_file)
    school_df = pd.read_json(school_file)
    student_df_raw = pd.read_csv(student_data_file, header=None) # Assume no header, rely purely on index
except Exception as e:
    st.error(f"Error loading data files: {e}")
    st.stop()

# Global weeks horizon (for model)
W = list(range(1, 200)) # 1..199

# --- CORE FIX: RENAME 'School' COLUMN TO 'School_Name' FOR CONSISTENCY (in JSON data) ---
if "School" in school_df.columns:
    school_df.rename(columns={"School": "School_Name"}, inplace=True)
elif "School_Name" not in school_df.columns:
     # Fallback if 'School' or 'School_Name' is missing
    school_df["School_Name"] = "School " + school_df["School_ID"].astype(str)

# Apply the same cleaning/normalization to the JSON school names for reliable matching
# FIX: Handle potential None/NaN values in the School_Name column from the JSON file
school_df['School_Name'] = school_df['School_Name'].apply(lambda x: x.upper().strip() if isinstance(x, str) else "")  
# -------------------------------------------------------------------------


# ---------------------------------------------------------
#        AUTO-DETECT POSTAL COLUMN
# ---------------------------------------------------------
possible_postal_cols = [
    "Postal_Code", "PostalCode", "postal_code",
    "Postal Code", "ZIP", "Zip", "PostCode"
]
postal_col = next((c for c in school_df.columns if c in possible_postal_cols), None)

if postal_col is None:
    st.error("Postal code column missing. Please ensure your school data has a recognized postal code column.")
    st.stop()

# ---------------------------------------------------------
#        SIDEBAR — CLINICIAN ID
# ---------------------------------------------------------
st.sidebar.header("🧑‍⚕️ Clinician Selection", divider="gray")

clinician_id_input = st.sidebar.number_input(
    "Enter Clinician ID for analysis",
    min_value=0,
    step=1,
    value=int(slp_df["Clinician_ID"].iloc[0]) if not slp_df.empty else 0,
    on_change=reset_optimization_state
)
clinician_id_input = int(clinician_id_input) # Ensure it's an integer

# ---------------------------------------------------------
#        DETERMINE WORKING DAYS (GLOBAL)
# ---------------------------------------------------------
# Define D (range of working days per week) based on schedule
try:
    clinician_schedule_row = slp_df.loc[slp_df["Clinician_ID"] == clinician_id_input]
    if clinician_schedule_row.empty:
        st.error(f"Clinician ID {clinician_id_input} not found in SLP data.")
        st.stop()
        
    clinician_schedule_str = clinician_schedule_row["Schedule"].squeeze()
    if pd.isna(clinician_schedule_str):
        # Fallback or error
        st.error("Clinician schedule not found.")
        st.stop()
except Exception as e:
    st.error(f"Invalid Clinician ID or data issue: {e}")
    st.stop()

schedule_days_map = {"m-f": 5, "tu-th": 3, "m-th": 4}
# 1=Mon, 2=Tue, 3=Wed, 4=Thu, 5=Fri
day_index_map = {"m-f": [1, 2, 3, 4, 5], "tu-th": [2, 3, 4], "m-th": [1, 2, 3, 4]}
weekday_names_map = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri"}

normalized_schedule = str(clinician_schedule_str).strip().lower().replace("–", "-").replace("m-th", "m-th")
if normalized_schedule not in day_index_map:
    # Fallback or stop
    st.warning(f"Schedule '{normalized_schedule}' not recognized. Defaulting to M-F.")
    normalized_schedule = "m-f"

num_days = schedule_days_map[normalized_schedule]
D = range(1, num_days + 1)
working_weekdays = day_index_map[normalized_schedule]

# =========================================================
# NEW LOGIC: CSV PROCESSING AND COUNT CALCULATION
# =========================================================

try:
    student_df = student_df_raw.copy()
    
    # 1. Check if the CSV has enough columns
    if len(student_df.columns) <= REFERRAL_DATE_COL_INDEX:
        st.error(f"CSV file must have at least {REFERRAL_DATE_COL_INDEX + 1} columns. Only found {len(student_df.columns)}.")
        st.stop()

    # Create descriptive names for the columns based on their fixed indices
    student_df.rename(columns={
        SCHOOL_NAME_COL_INDEX: 'CSV_SchoolName',
        PRIORITY_COL_INDEX: 'CSV_Priority',
        REFERRAL_DATE_COL_INDEX: 'CSV_ReferralDate'
    }, inplace=True)
    
    # 2. Clean school names and filter/clean priority
    student_df['Clean_School_Name'] = student_df['CSV_SchoolName'].apply(clean_school_name)
    student_df['Priority_Code'] = pd.to_numeric(student_df['CSV_Priority'], errors='coerce')
    
    # Filter for valid students: 1 (HP) or 2 (LP)
    valid_students = student_df[student_df['Priority_Code'].isin([1, 2])].copy()
    
    # 3. Calculate HP (1) and LP (2) counts
    hp_counts = valid_students[valid_students['Priority_Code'] == 1].groupby('Clean_School_Name').size().rename('Total_HP')
    lp_counts = valid_students[valid_students['Priority_Code'] == 2].groupby('Clean_School_Name').size().rename('Total_LP')

    # Merge counts into a summary table
    schools_with_demand = pd.DataFrame({'School_Name': hp_counts.index.union(lp_counts.index)})
    schools_with_demand.rename(columns={'School_Name': 'Clean_School_Name'}, inplace=True) 
    schools_with_demand = schools_with_demand.merge(hp_counts, on='Clean_School_Name', how='left').fillna(0)
    schools_with_demand = schools_with_demand.merge(lp_counts, on='Clean_School_Name', how='left').fillna(0)
    schools_with_demand['Demand'] = schools_with_demand['Total_HP'] + schools_with_demand['Total_LP']
    
    # Rename back to 'School_Name' for merging
    schools_with_demand.rename(columns={'Clean_School_Name': 'School_Name'}, inplace=True)

    # Filter for schools with actual demand > 0
    schools_with_demand = schools_with_demand[schools_with_demand['Demand'] > 0].reset_index(drop=True)
    for col in ['Total_HP', 'Total_LP', 'Demand']:
        schools_with_demand[col] = schools_with_demand[col].astype(int)

except Exception as e:
    st.error(f"Error processing CSV columns. Please verify the CSV format and column indices. Error: {e}")
    st.stop()


# --- Match with Clinician's assigned schools ---
assigned_schools_raw = school_df[
    school_df["SLP_IDs"].apply(
        lambda ids: clinician_id_input in ids if isinstance(ids, list) else False
    )
].copy()

# Filter assigned schools by those with demand AND assigned to the clinician
assigned_schools_with_demand = assigned_schools_raw[['School_Name', 'School_ID', 'SLP_IDs']].merge(
    schools_with_demand,
    on='School_Name', 
    how='inner' 
).reset_index(drop=True)

if assigned_schools_with_demand.empty:
    st.error(f"Clinician **{clinician_id_input}** has no assigned schools with positive demand from the CSV data. Please check clinician assignment or student data.")
    st.stop()

# ---------------------------------------------------------
#        FINAL COUNTS DERIVATION (INTO EDITABLE SESSION STATE)
# ---------------------------------------------------------

# 1. Calculate Default Counts based on the current clinician input
default_counts_df = assigned_schools_with_demand[['School_Name', 'School_ID', 'Total_HP', 'Total_LP', 'Demand']].copy()
# Default split: 20% HI
default_counts_df['HI_HP'] = (default_counts_df['Total_HP'] * 0.2).astype(int)
default_counts_df['HI_LP'] = (default_counts_df['Total_LP'] * 0.2).astype(int)
# Cap at total
default_counts_df['HI_HP'] = np.minimum(default_counts_df['HI_HP'], default_counts_df['Total_HP']).astype(int)
default_counts_df['HI_LP'] = np.minimum(default_counts_df['HI_LP'], default_counts_df['Total_LP']).astype(int)
# Derived LI
default_counts_df['LI_HP'] = np.maximum(0, default_counts_df['Total_HP'] - default_counts_df['HI_HP']).astype(int)
default_counts_df['LI_LP'] = np.maximum(0, default_counts_df['Total_LP'] - default_counts_df['HI_LP']).astype(int)

# 2. Check if we need to initialize or reset the session state data
# Reset if it's the first run OR if the clinician ID changed since the last run
if st.session_state.editable_counts is None or st.session_state.last_clinician_id != clinician_id_input:
    st.session_state.editable_counts = default_counts_df.copy()
    st.session_state.last_clinician_id = clinician_id_input

# --- SIDEBAR DISPLAY TABLE (FULLY EDITABLE) ---
st.sidebar.header("📊 Student Counts by Category (Editable)", divider="gray")
st.sidebar.markdown(
    """
    Counts derived from the CSV. **Edit any of the 4 category columns** (HI-HP, HI-LP, LI-HP, LI-LP). 
    Totals are fixed from the CSV data. Please ensure your categories sum correctly if you wish to match the original total.
    """
)

# Configuration for columns
# FIXED: Total demand columns are disabled. All 4 category columns are enabled.
column_config = {
    'School_Name': st.column_config.TextColumn("School Name", disabled=True),
    'School_ID': st.column_config.NumberColumn("ID", disabled=True, width="small"),
    'Demand': st.column_config.NumberColumn("Total (CSV)", disabled=True, width="small", help="Fixed Total from CSV"),
    'Total_HP': st.column_config.NumberColumn("Tot HP (CSV)", disabled=True, width="small"),
    'Total_LP': st.column_config.NumberColumn("Tot LP (CSV)", disabled=True, width="small"),
    # Editable Columns - User has total freedom
    'HI_HP': st.column_config.NumberColumn("HI-HP", min_value=0, step=1, required=True),
    'HI_LP': st.column_config.NumberColumn("HI-LP", min_value=0, step=1, required=True),
    'LI_HP': st.column_config.NumberColumn("LI-HP", min_value=0, step=1, required=True),
    'LI_LP': st.column_config.NumberColumn("LI-LP", min_value=0, step=1, required=True),
}

# 3. Display Data Editor and capture edits
# We pass the Session State DF. The return value is the *edited* dataframe.
edited_counts_df = st.sidebar.data_editor(
    st.session_state.editable_counts,
    column_config=column_config,
    column_order=['School_Name', 'Demand', 'Total_HP', 'Total_LP', 'LI_LP', 'LI_HP', 'HI_LP', 'HI_HP'],
    use_container_width=True,
    hide_index=True,
    key="school_counts_editor" 
)

# 4. Persist: Update Session State with the edited DF
# We do NOT overwrite LI values automatically anymore, allowing the user full control.
cols_to_int = ['Total_HP', 'Total_LP', 'Demand', 'HI_HP', 'HI_LP', 'LI_HP', 'LI_LP']
edited_counts_df[cols_to_int] = edited_counts_df[cols_to_int].astype(int)
st.session_state.editable_counts = edited_counts_df.copy()

# 5. Use the Edited Data for the rest of the app
final_counts_df = st.session_state.editable_counts.copy()


# Update the master school_df for display consistency elsewhere
for _, row in final_counts_df.iterrows():
    sname = row['School_Name']
    total_demand = row["Total_HP"] + row["Total_LP"]
    mask = (school_df["School_Name"] == sname) & (
        school_df["SLP_IDs"].apply(lambda ids: clinician_id_input in ids if isinstance(ids, list) else False)
    )
    school_df.loc[mask, ["LI_LP", "LI_HP", "HI_LP", "HI_HP", "Demand"]] = (
        row["LI_LP"], row["LI_HP"], row["HI_LP"], row["HI_HP"], total_demand
    )


# ---------------------------------------------------------
#        SIDEBAR — SCHOOL SELECTION (Filtered by Demand > 0)
# ---------------------------------------------------------
st.sidebar.header("🏫 School Selection", divider="gray")

available_school_names = assigned_schools_with_demand["School_Name"].tolist()

# Select top 8 schools by demand as default for model input
top8_names = (
    assigned_schools_with_demand.sort_values("Demand", ascending=False)
    .head(8)["School_Name"].tolist()
)

selected_school_names = st.sidebar.multiselect(
    "Select schools for the scheduling problem",
    options=available_school_names,
    default=top8_names
)

# Convert selected names back to IDs for filtering
name_to_id = assigned_schools_with_demand.set_index("School_Name")["School_ID"].to_dict()
selected_school_ids = [name_to_id[name] for name in selected_school_names]

# Get the list of assigned schools that are SELECTED by the user, with the updated counts
# THIS USES THE EDITED final_counts_df
assigned_schools = final_counts_df[
    final_counts_df["School_ID"].isin(selected_school_ids)
].reset_index(drop=True)

if assigned_schools.empty:
    st.info("Select at least one school with demand to proceed to optimization inputs.")
    st.stop()

# ---------------------------------------------------------
# Define num_schools and schools list based on the final selection
# ---------------------------------------------------------
num_schools = len(assigned_schools)
schools = list(range(num_schools))
# ---------------------------------------------------------

# ---------------------------------------------------------
#        SIDEBAR — PARAMETERS
# ---------------------------------------------------------
st.sidebar.header("⚙️ Optimization Parameters", divider="gray")
st.sidebar.markdown("**Session Requirements**")
n1_input = st.sidebar.number_input("Sessions Required (Non-HITS, Low Priority)", min_value=1, value=8, step=1)
n2_input = st.sidebar.number_input("Sessions Required (Non-Hits, High Priority)", min_value=1, value=10, step=1)
lp_ratio = st.sidebar.number_input("Ratio HITS to Non-HITS (Low Priority)", min_value=0.0, value=2.0, step=0.1)
hp_ratio = st.sidebar.number_input("Ratio HITS to Non-HITS (High Priority)", min_value=0.0, value=2.0, step=0.1)

st.sidebar.markdown("**Capacity & Duration**")
cap_total = st.sidebar.number_input("Weekly Capacity (Sessions)", min_value=1, value=40, step=1)
theta_low = st.sidebar.number_input("Duration Non-HITS (min)", min_value=1, value=40, step=1)
theta_high = st.sidebar.number_input("Duration HITS (min)", min_value=1, value=7, step=1)
theta_max = st.sidebar.number_input("Max Minutes per Day", min_value=1, value=5 * 60, step=5) # 5 hours = 300 minutes

st.sidebar.markdown("**Priority Weight**")
priority_ratio_c = st.sidebar.number_input("Priority Weight (High vs Low)", min_value=1.0, value=5.0, step=0.5)

st.sidebar.markdown("**Planning Horizon**")
planning_horizon = st.sidebar.number_input("Horizon (Weeks)", min_value=1, max_value=len(W), value=16, step=1)
target_week = list(range(1, planning_horizon + 1))


# =========================================================
# COORDINATE LOOKUP & TRAVEL MATRIX SETUP
# =========================================================
# --- Coordinate Lookup Function ---
nomi = pgeocode.Nominatim("CA")
geo = Nominatim(user_agent="clinician_mapping")
coord_cache = {}

@st.cache_data(show_spinner=False)
def get_coordinates_cached(pc):
    if not pc:
        return (None, None)
    pc = str(pc).upper().replace(" ", "")
    if len(pc) == 6:
        pc = pc[:3] + " " + pc[3:]
    
    if pc in coord_cache:
        return coord_cache[pc]
        
    rec = nomi.query_postal_code(pc)
    if rec is not None and not pd.isna(rec.latitude):
        coord_cache[pc] = (float(rec.latitude), float(rec.longitude))
        return coord_cache[pc]
    try:
        loc = geo.geocode({"postalcode": pc, "country": "Canada"})
        time.sleep(1)
        if loc:
            coord_cache[pc] = (loc.latitude, loc.longitude)
            return coord_cache[pc]
    except:
        pass
    coord_cache[pc] = (None, None)
    return (None, None)

# --- Apply Coordinates to school_df (Master DF) ---
# This is crucial for retrieving coordinates later.
if "Coordinates" not in school_df.columns or school_df["Coordinates"].isna().any():
    school_df["Coordinates"] = school_df[postal_col].apply(get_coordinates_cached)

# --- Apply Coordinates to assigned_schools (Model DF) ---
# This DataFrame is used for the optimization model setup (time matrix, student data)
assigned_schools = assigned_schools.merge(
    school_df[['School_ID', 'Coordinates']], on='School_ID', how='left'
)
assigned_schools.dropna(subset=['Coordinates'], inplace=True) 

# --- Travel Matrix Setup ---
def euclidean_distance(coord1, coord2):
    if coord1 is None or coord2 is None or None in coord1 or None in coord2:
        return np.nan
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    # Simple conversion factor approximation for a smaller region
    dlat = (lat2 - lat1) * 111 
    dlon = (lon2 - lon1) * 85
    return sqrt(dlat**2 + dlon**2)

AVERAGE_SPEED = 40  # km/h
KM_TO_MIN = 60 / AVERAGE_SPEED

coords = assigned_schools["Coordinates"].tolist()

# The model uses 0 for Depot/Office, and 1 to N for Schools
time_matrix = np.zeros((num_schools + 1, num_schools + 1))
for i in range(num_schools):
    for j in range(num_schools):
        if i != j:
            dist_km = euclidean_distance(coords[i], coords[j])
            time_matrix[i + 1, j + 1] = round(dist_km * KM_TO_MIN, 1)
for j in range(1, num_schools + 1):
    # Travel time from/to depot is assumed to be 0 for simplicity in this abstract model setup
    time_matrix[0, j] = 0
    time_matrix[j, 0] = 0
    
school_names = assigned_schools.set_index(assigned_schools.index + 1)['School_Name'].to_dict()
school_names[0] = "Office/Depot"
depot_lat = assigned_schools["Coordinates"].apply(lambda c: c[0]).mean()
depot_lon = assigned_schools["Coordinates"].apply(lambda c: c[1]).mean()
depot_coords = (depot_lat, depot_lon)

nodes = range(num_schools + 1)  # 0 = depot, 1..|schools|
s_idx = range(1, num_schools + 1) # <--- DEFINED HERE


# =========================================================
# NEW LOGIC: BUILD STUDENTS FROM CSV COUNTS (USING EDITED COUNTS)
# =========================================================
np.random.seed(42)

# Start building temporary student structures (t_ for temporary)
t_student_school = {}
t_p = {}  # priority
t_I = {}  # intensity
t_r = {}  # referral week
t_students = []  # track valid student IDs
school_type_counts = {}

# Map for student creation: School_Name -> DataRow from CSV
student_ref_dates_by_school = valid_students[['Clean_School_Name', 'Priority_Code', 'CSV_ReferralDate']].copy()
student_ref_dates_by_school['r_val'] = student_ref_dates_by_school['CSV_ReferralDate'].apply(lambda x: calculate_r_value(x, ANCHOR_DATE))
student_ref_dates_by_school.dropna(subset=['r_val'], inplace=True)

# Student ID generator
idx = 0 
for j, row in assigned_schools.iterrows():

    s_name = row["School_Name"]
    # CRUCIAL: Use the counts from the assigned_schools DataFrame (which reflects user edits)
    li_lp = int(row["LI_LP"])
    li_hp = int(row["LI_HP"])
    hi_lp = int(row["HI_LP"])
    hi_hp = int(row["HI_HP"])
    
    # Store for pie charts
    school_type_counts[j] = {
        "f1": li_lp, "f2": li_hp, "f3": hi_lp, "f4": hi_hp
    }

    # Retrieve all student records (rows from CSV) for this school
    school_students_data = student_ref_dates_by_school[
        student_ref_dates_by_school['Clean_School_Name'] == s_name
    ].copy()

    # Separate students by priority and sort by referral week (oldest first)
    lp_available = school_students_data[school_students_data['Priority_Code'] == 2].sort_values(by='r_val', ascending=True)
    hp_available = school_students_data[school_students_data['Priority_Code'] == 1].sort_values(by='r_val', ascending=True)

    # Convert to list of r_vals to be consumed
    lp_pool = lp_available[['r_val']].values.tolist()
    hp_pool = hp_available[['r_val']].values.tolist()

    # --- Integrated Student Assignment Logic (Takes the oldest students first) ---
    
    # HI-HP (p=1, I=1): Take 'hi_hp' oldest HP students
    for _ in range(hi_hp):
        if not hp_pool: break
        r_val = hp_pool.pop(0)[0]
        t_student_school[idx] = j + 1 # OFFSET SCHOOL INDEX BY +1 TO MATCH S_IDX (1-based)
        t_p[idx] = 1
        t_I[idx] = 1
        t_r[idx] = r_val
        t_students.append(idx)
        idx += 1

    # HI-LP (p=2, I=1): Take 'hi_lp' oldest LP students
    for _ in range(hi_lp):
        if not lp_pool: break
        r_val = lp_pool.pop(0)[0]
        t_student_school[idx] = j + 1 # OFFSET SCHOOL INDEX BY +1 TO MATCH S_IDX (1-based)
        t_p[idx] = 2
        t_I[idx] = 1
        t_r[idx] = r_val
        t_students.append(idx)
        idx += 1
    
    # LI-HP (p=1, I=0): The rest of the remaining HP students
    for _ in range(li_hp):
        if not hp_pool: break
        r_val = hp_pool.pop(0)[0]
        t_student_school[idx] = j + 1 # OFFSET SCHOOL INDEX BY +1 TO MATCH S_IDX (1-based)
        t_p[idx] = 1
        t_I[idx] = 0
        t_r[idx] = r_val
        t_students.append(idx)
        idx += 1
    
    # LI-LP (p=2, I=0): The rest of the remaining LP students
    for _ in range(li_lp):
        if not lp_pool: break
        r_val = lp_pool.pop(0)[0]
        t_student_school[idx] = j + 1 # OFFSET SCHOOL INDEX BY +1 TO MATCH S_IDX (1-based)
        t_p[idx] = 2
        t_I[idx] = 0
        t_r[idx] = r_val
        t_students.append(idx)
        idx += 1

# FINAL SANITIZATION: Re-index to ensure contiguous IDs from 0 to N-1
# We map the old indices to the new contiguous indices
id_map = {old_id: new_id for new_id, old_id in enumerate(t_students)}

students = list(range(len(t_students))) 
student_school = {id_map[i]: t_student_school[i] for i in t_students}
p = {id_map[i]: t_p[i] for i in t_students}
I = {id_map[i]: t_I[i] for i in t_students}
r = {id_map[i]: t_r[i] for i in t_students}


if len(students) == 0:
    st.error("ERROR: No students were generated for the selected schools. Check your inputs.")
    st.stop()
    
# ---------------------------------------------------------
# Rerunning parameter calculation for the final student set
# ---------------------------------------------------------
valid_student_ids = students

# Use the final dictionaries for calculation
n1 = int(n1_input)
n2 = int(n2_input)
n3 = int(lp_ratio * n1)
n4 = int(hp_ratio * n2)

theta = {}
sessions_needed = {}
weeks_needed = {}
c = {}

for i in students:
    theta[i] = theta_high if I[i] == 1 else theta_low
    
    if I[i] == 0:
        # low intensity
        sessions_needed[i] = n1 if p[i] == 2 else n2
        weeks_needed[i] = sessions_needed[i] * 2
    else:
        # high intensity
        sessions_needed[i] = n3 if p[i] == 2 else n4
        weeks_needed[i] = sessions_needed[i] // 3
        if sessions_needed[i] > 0 and weeks_needed[i] == 0: weeks_needed[i] = 1
        if sessions_needed[i] % 3 != 0: weeks_needed[i] += 1
            
    c[i] = priority_ratio_c if p[i] == 1 else 1.0


# ============================================================
# START OF MAIN PAGE CONTENT (REORDERED & ENHANCED)
# ============================================================

# --- FIX: Ensure assigned_schools_with_demand has Coordinates for the initial map display ---
if "Coordinates" in school_df.columns:
    # Merge coordinates into the data used for the main display map
    assigned_schools_with_demand = assigned_schools_with_demand.merge(
        school_df[['School_ID', 'Coordinates']], 
        on='School_ID', 
        how='left'
    )
    
## Clinician Overview 🧑‍⚕️
st.header("Clinician Overview and Assignments", divider="gray")
st.markdown(
    """
    Now enter the clinician ID for the clinician whose optimal schedule you would like to view. To protect privacy, clinician names are hidden and each clinician is assigned an ID instead. After entering the ID, a map will appear showing the schools assigned to that clinician. On the left, you can also view and adjust each school's current waitlist.
    """
)

clinician = slp_df[slp_df["Clinician_ID"] == clinician_id_input]
clinician_row = slp_df[slp_df["Clinician_ID"] == clinician_id_input]

if clinician.empty:
    st.info("Enter a valid clinician ID.")
    st.stop()

schedule = clinician_row["Schedule"].iloc[0]

col_info, col_map = st.columns([1, 2])

with col_info:
    st.subheader(f"ID: **{clinician_id_input}**")
    st.metric(label="Working Schedule", value=f"**{schedule}** ({num_days} days/week)")
    st.metric(label="Assigned Schools (with Demand)", value=len(assigned_schools_with_demand))
    
    st.markdown(
        f"""
        <div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; margin-top: 15px;'>
            **Planning Details:**
            <ul>
                <li>**Total Students (Optimized Set):** {len(students)}</li>
                <li>**Planning Horizon (Weeks):** {planning_horizon}</li>
                <li>**Anchor Date (Week 1):** {ANCHOR_DATE}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True
    )
    
    with st.expander("Show All Assigned Schools"):
        assigned_schools_display = school_df[
            school_df["SLP_IDs"].apply(lambda ids: clinician_id_input in ids if isinstance(ids, list) else False)
        ].copy()
        
        for _, row in assigned_schools_display.iterrows():
            demand_info = f" ({int(row.get('Demand', 0))} Students)" if 'Demand' in row and row['Demand'] > 0 else ""
            st.write(f"• {row['School_Name']} (ID: {row['School_ID']}){demand_info}")  

with col_map:
    st.subheader("School Locations and Service Area")
    
    # Use assigned_schools_with_demand for valid schools on the map
    valid_schools = assigned_schools_with_demand[
        assigned_schools_with_demand["Coordinates"].apply(lambda c: c and c[0] is not None)
    ].copy()

    if valid_schools.empty:
        st.warning("No valid coordinates found for the assigned schools.")
    else:
        center_lat = valid_schools["Coordinates"].apply(lambda c: c[0]).mean()
        center_lon = valid_schools["Coordinates"].apply(lambda c: c[1]).mean()

        fmap = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="OpenStreetMap")
        
        # Add Depot (Centroid) Marker
        folium.Marker(
            depot_coords,
            icon=folium.Icon(color='green', icon='briefcase', prefix='fa'),
            popup=f"**Office/Depot**"
        ).add_to(fmap)

        for _, row in valid_schools.iterrows():
            lat, lon = row["Coordinates"]
            sname = row["School_Name"]
            demand_val = int(row["Demand"])

            # USE UNIFORM COLOR
            color = COLOR_SCHOOL_UNIFORM
            radius = 5 + (demand_val / valid_schools['Demand'].max()) * 10
            
            # School Marker (Circle)
            folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                fill=True,
                color=color,
                fill_color=color,
                fill_opacity=0.7,
                popup=f"**{sname}**<br>Demand: **{demand_val}** students"
            ).add_to(fmap)
            
            # School Name Label (DivIcon)
            folium.Marker(
                [lat, lon],
                icon=folium.DivIcon(
                    html=f'<div style="font-size: 10pt; color: {color}; font-weight: bold; transform: translate(15px, -10px);">{sname}</div>'
                )
            ).add_to(fmap)


        st_folium(fmap, width=None, height=450)

st.markdown("---")

## School-Level Demand Analysis 🏫
st.header("School-Level Demand Analysis", divider="gray")
st.markdown(
    """
This section details the composition of student demand for the **selected schools** in the sidebar. Note that 'LI' referrs to Low intensity (normal sessions), 'HI' referrs to HITS sessions, 'LP' referrs to low priority (priority 2), and 'HP' referrs to high priority (priority 1)
"""
)

# Define pie chart colors
pie_colors = [COLOR_LI_LP, COLOR_LI_HP, COLOR_HI_LP, COLOR_HI_HP]
labels = ["LI-LP", "LI-HP", "HI-LP", "HI-HP"]

if num_schools == 0:
    st.info("No schools available for school information. Please select schools with demand.")
else:
    # Create a dynamic number of columns for better display
    num_cols = min(4, num_schools)
    cols = st.columns(num_cols)

    for j in schools:
        # Use the *assigned_schools* which contains the edited counts
        school_name = assigned_schools.loc[j, 'School_Name']
        
        counts = {
            "f1": int(assigned_schools.loc[j, "LI_LP"]),
            "f2": int(assigned_schools.loc[j, "LI_HP"]),
            "f3": int(assigned_schools.loc[j, "HI_LP"]),
            "f4": int(assigned_schools.loc[j, "HI_HP"])
        }
        
        sizes = [counts["f1"], counts["f2"], counts["f3"], counts["f4"]]
        total = sum(sizes)

        if total == 0:
            continue

        col = cols[j % num_cols]
        with col:
            fig, ax = plt.subplots(figsize=(2.5, 2.5))
            ax.pie(
                sizes,
                labels=None, # Remove labels on the slices for cleaner look
                colors=pie_colors, 
                autopct=lambda p: '{:.0f}'.format(p * total / 100) if p > 0 else '', # Display absolute number if > 0
                pctdistance=0.7,
                wedgeprops={'edgecolor': 'black', 'linewidth': 0.5},
            )
            ax.set_title(
                f"{school_name}\n({total} Students)",
                fontsize=10, weight='bold'
            )
            ax.legend(labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=7)
            st.pyplot(fig)


# ---- r[i] plots by priority and intensity ----
st.subheader("Student Referral Week Distribution")
st.markdown(
    f"""
The value of the referral week represents the student's referral time in weeks, calculated based on the date in the CSV file relative to the **Anchor Date: {ANCHOR_DATE} (Week 1)**. A smaller value means an older referral. Note that e.g. if the referral date is 2025-12-01, i.e., Dec. 1, 2025, then the referral week is -4, since it is calculated with respect to Jan. 1, 2026.
"""
)

if len(students) == 0:
    st.info("No students to plot $r_i$.")
else:
    col1, col2 = st.columns(2)

    # Plot 1: colored by priority (p[i])
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        xs = []
        ys = []
        colors_pr = []
        for i in students:
            xs.append(i)
            ys.append(r[i])
            colors_pr.append(COLOR_HI_HP if p[i] == 1 else COLOR_LI_LP) 
        ax1.scatter(xs, ys, c=colors_pr, s=15, alpha=0.7)
        ax1.scatter([], [], color=COLOR_HI_HP, label="High Priority (p=1)")
        ax1.scatter([], [], color=COLOR_LI_LP, label="Low Priority (p=2)")
        ax1.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax1.legend()
        ax1.set_title("Referral Week by Student Priority")
        ax1.set_xlabel("Student Index")
        ax1.set_ylabel("Referral Week ($r_i$)")
        st.pyplot(fig1)

    # Plot 2: colored by intensity (I[i])
    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        xs2 = []
        ys2 = []
        colors_int = []
        for i in students:
            xs2.append(i)
            ys2.append(r[i])
            colors_int.append(COLOR_HI_HP if I[i] == 1 else COLOR_LI_LP) 
        ax2.scatter(xs2, ys2, c=colors_int, s=15, alpha=0.7)
        ax2.scatter([], [], color=COLOR_HI_HP, label="High Intensity (I=1)")
        ax2.scatter([], [], color=COLOR_LI_LP, label="Low Intensity (I=0)")
        ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax2.legend()
        ax2.set_title("Referral Week by Student Intensity")
        ax2.set_xlabel("Student Index")
        ax2.set_ylabel("Referral Week ($r_i$)")
        st.pyplot(fig2)

st.markdown("---")

## Optimization Control and Status ⚙️
st.header("Optimization Model", divider="gray")
st.markdown(
    """
    Enter the parameters for the model:

    1. <strong style="color: blue;">School Selection</strong> Select the schools for which you want to solve the scheduling problem.  
       By default, the app selects the eight most populated schools, but you may change this manually.  
       *Note: selecting more than eight schools may increase solving time.*

    2. <strong style="color: blue;">Number of Sessions per Student Type</strong> Enter the number of sessions required for:
       - Normal (non-HITS) students — both low-priority and high-priority  
       - HITS students — based on two learning-curve ratios (low-priority and high-priority)

    3. <strong style="color: blue;">Maximum Weekly Sessions</strong> Input the maximum number of total sessions the clinician can handle each week.  
       The optimizer will not exceed this limit.

    4. <strong style="color: blue;">Session Durations</strong> Specify the duration of:  
       - **Non-HITS sessions** (default: 40 minutes = 30 session + 10 documentation)  
       - **HITS sessions** (default: 7 minutes = 5 session + 2 documentation)

    5. <strong style="color: blue;">Daily Working Time</strong> Enter the clinician’s daily available working minutes (default: 300 minutes = 5 hours).

    6. <strong style="color: blue;">Priority Ratio</strong> Provide the weighting factor for high-priority students (both HITS and non-HITS),  
       which determines how strongly the model prioritizes them.

    7. <strong style="color: blue;">Planning Horizon</strong> Provide the planning horizon you wish the model to be solved for. (note that usually, not all students can be served in that horizon)
    """,
    unsafe_allow_html=True
)
st.info("Ready to run. Adjust parameters in the sidebar and click **'Run Optimization'**.")

status_placeholder = st.empty()
progress_bar = st.progress(0)

# --- Optimization Run Logic (No change to logic, just wrapped) ---
start_optimization = st.sidebar.button("Run Optimization")

if start_optimization:
    # Reset success state when optimization starts
    st.session_state.optimization_success = False 
    try:
        st.session_state.optimization_ran = True

        status_placeholder.write("Starting **Weekly Master Problem**...")
        progress_bar.progress(5)

        weekly = gp.Model(f"clinician_{clinician_id_input}_weekly") # FIX: Used clinician_id_input

        weekly.Params.MIPGap = 0.1
        weekly.Params.TimeLimit = 1000 * 60
        weekly.Params.OutputFlag = 0

        x_li_var = weekly.addVars(students, W, vtype=GRB.BINARY, name="x_li")
        aux1_var = weekly.addVars(students, W, vtype=GRB.BINARY, name="aux1")
        y_var  = weekly.addVars(students, W, vtype=GRB.BINARY, name="y")
        tr_var   = weekly.addVars(s_idx, W, vtype=GRB.BINARY, name="tr")

        # 1) unique start
        for i in students:
            weekly.addConstr(
                gp.quicksum(y_var[i, w] for w in W) == 1,
                name=f"unique_start_i{i}"
            )

        # 2) low intensity continuity
        for i in students:
            if I[i] == 0:
                for w in W:
                    for k in range(sessions_needed[i]):
                        wk = w + 2 * k
                        if wk in W:
                            weekly.addConstr(
                                x_li_var[i, wk] >= y_var[i, w],
                                name=f"LI_block_LB[{i},{w},{wk}]"
                            )
                for u in W:
                    weekly.addConstr(
                        x_li_var[i, u] <= gp.quicksum(
                            y_var[i, u - 2 * k]
                            for k in range(sessions_needed[i])
                            if (u - 2 * k) in W
                        ),
                        name=f"LI_block_UB[{i},{u}]"
                    )

        # 3) high intensity continuity
        for i in students:
            if I[i] == 1:
                WN = weeks_needed[i]
                for w in W:
                    for k in range(WN):
                        wk = w + k
                        if wk in W:
                            weekly.addConstr(
                                aux1_var[i, wk] >= y_var[i, w],
                                name=f"HI_block_LB[{i},{w},{wk}]"
                            )
                for u in W:
                    weekly.addConstr(
                        aux1_var[i, u] <= gp.quicksum(
                            y_var[i, u - k]
                            for k in range(WN)
                            if (u - k) in W
                        ),
                        name=f"HI_block_UB[{i},{u}]"
                    )

        # 4) weekly capacity
        cap_total_value = cap_total
        cap_total_constraints = {}
        for w in W:
            expr = (3 * gp.quicksum(aux1_var[i, w] for i in students if I[i] == 1) +
                    1 * gp.quicksum(x_li_var[i, w] for i in students if I[i] == 0))
            cap_total_constraints[w] = weekly.addConstr(
                expr <= cap_total_value,
                name=f"cap_total[{w}]"
            )

        # 5) travel link
        for i in students:
            j = student_school[i]
            for w in W:
                if I[i] == 0:
                    weekly.addConstr(
                        tr_var[j, w] >= x_li_var[i, w],
                        name=f"tr_ge_xli_i{i}_j{j}_w{w}"
                    )
                else:
                    weekly.addConstr(
                        tr_var[j, w] >= aux1_var[i, w],
                        name=f"tr_ge_aux1_i{i}_j{j}_w{w}"
                    )

        # 6) max schools per week
        max_schools_per_week_value = 4
        max_schools_constraints = {}
        for w in W:
            expr = gp.quicksum(tr_var[j, w] for j in s_idx)
            max_schools_constraints[w] = weekly.addConstr(
                expr <= max_schools_per_week_value,
                name=f"max_schools_per_week_w{w}"
            )

        # 7) no travel if no visits
        for j in s_idx:
            for w in W:
                total_visits = gp.quicksum(
                    (x_li_var[i, w] if I[i] == 0 else aux1_var[i, w])
                    for i in students if student_school[i] == j
                )
                weekly.addConstr(
                    tr_var[j, w] <= total_visits,
                    name=f"no_travel_without_visit_j{j}_w{w}"
                )

        weekly.setObjective(
            gp.quicksum(
                c[i] * (gp.quicksum(w * y_var[i, w] for w in W) - r[i])
                for i in students
            ),
            GRB.MINIMIZE
        )

        MAX_ITER = 50
        CAP_DECREASE = 1
        SCHOOLS_DECREASE = 1
        MIN_CAP_TOTAL = 0
        MIN_SCHOOLS_PER_WEEK = 1

        for it in range(MAX_ITER):
            progress = min(5 + int(60 * it / MAX_ITER), 70)
            progress_bar.progress(progress)
            status_placeholder.write(
                f"**Weekly Master Problem** (Iteration {it}): Solving "
                f"(Current Weekly Cap={cap_total_value}, Max Schools/Week={max_schools_per_week_value})..."
            )

            weekly.optimize()

            if weekly.status != GRB.OPTIMAL:
                status_placeholder.error("Weekly model infeasible or not optimal. Stopping.")
                break

            # Store weekly results to session state
            st.session_state.x_li_weekly = {(i, w): int(round(x_li_var[i, w].X)) for i in students for w in W}
            st.session_state.aux1_weekly = {(i, w): 3 * int(round(aux1_var[i, w].X)) for i in students for w in W}
            st.session_state.tr_weekly = {(j, w): int(round(tr_var[j, w].X)) for j in s_idx for w in W}

            status_placeholder.write("Weekly model solved. Starting **Daily Subproblem** (VRP) for routing feasibility...")
            progress_bar.progress(min(progress + 5, 80))

            m_daily = gp.Model(f"clinician_{clinician_id_input}_daily_scheduling") # FIX: Used clinician_id_input
            m_daily.Params.OutputFlag = 0
            m_daily.Params.TimeLimit = 60 # Set a max time limit for daily VRP to prevent excessive runtimes

            x_d  = m_daily.addVars(students, W, D, vtype=GRB.BINARY, name="x")
            v    = m_daily.addVars(s_idx, W, D, vtype=GRB.BINARY, name="v")
            z    = m_daily.addVars(nodes, nodes, W, D, vtype=GRB.BINARY, name="z")
            d_day = m_daily.addVars(W, D, vtype=GRB.BINARY, name="day_on")
            
            # --- Link constraints using session state data ---
            for i in students:
                for w in W:
                    if I[i] == 0:
                        m_daily.addConstr(
                            gp.quicksum(x_d[i, w, t] for t in D) == st.session_state.x_li_weekly[i, w],
                            name=f"link_low_i{i}_w{w}"
                        )
                    else:
                        m_daily.addConstr(
                            gp.quicksum(x_d[i, w, t] for t in D) == st.session_state.aux1_weekly[i, w] / 3, # Divide by 3 because aux1_weekly is weighted sessions
                            name=f"link_high_i{i}_w{w}"
                        )

            for i in students:
                j = student_school[i]
                for w in W:
                    for t in D:
                        if I[i] == 0:
                            m_daily.addConstr(
                                x_d[i, w, t] <= st.session_state.x_li_weekly[i, w],
                                name=f"ub_low_i{i}_w{w}_t{t}"
                            )
                            
                        # Link daily sessions to weekly school travel decision (tr_weekly)
                        m_daily.addConstr(
                            x_d[i, w, t] <= st.session_state.tr_weekly[j, w],
                            name=f"no_x_if_no_travel_i{i}_w{w}_t{t}"
                        )
            # --- End Link constraints ---


            # Daily Routing Constraints
            for w in W:
                for t in D:
                    # Must start and end at depot (0) if day is on
                    m_daily.addConstr(
                        gp.quicksum(z[0, k, w, t] for k in nodes if k != 0) == d_day[w, t],
                        name=f"start_from_depot_w{w}_t{t}"
                    )
                    m_daily.addConstr(
                        gp.quicksum(z[jj, 0, w, t] for jj in nodes if jj != 0) == d_day[w, t],
                        name=f"return_to_depot_w{w}_t{t}"
                    )

            for w in W:
                for t in D:
                    for j in s_idx:
                        # Flow balance: flow in = flow out for intermediate nodes (schools)
                        m_daily.addConstr(
                            gp.quicksum(z[k, j, w, t] for k in nodes if k != j) ==
                            gp.quicksum(z[j, k, w, t] for k in nodes if k != j),
                            name=f"flow_balance_j{j}_w{w}_t{t}"
                        )

            # Link visit variable v[j,w,t] to weekly travel variable tr[j,w]
            for j in s_idx:
                for w in W:
                    for t in D:
                        # Visit v can only happen if weekly travel tr is planned
                        m_daily.addConstr(
                            v[j, w, t] <= st.session_state.tr_weekly[j, w],
                            name=f"no_v_if_no_travel_j{j}_w{w}_t{t}"
                        )
                    # Visit must happen at least one day if weekly travel tr is planned
                    m_daily.addConstr(
                        gp.quicksum(v[j, w, t] for t in D) >= st.session_state.tr_weekly[j, w],
                        name=f"at_least_one_visit_j{j}_w{w}"
                    )

            # Link daily session x_d[i,w,t] to daily visit v[j,w,t]
            for i in students:
                j = student_school[i]
                for w in W:
                    for t in D:
                        # If a student i (at school j) has a session, school j must be visited
                        m_daily.addConstr(
                            v[j, w, t] >= x_d[i, w, t],
                            name=f"student_visit_link_i{i}_j{j}_w{w}_t{t}"
                        )

            # Link daily visit v[j,w,t] to routing variable z[j,k,w,t]
            for w in W:
                for t in D:
                    for j in s_idx:
                        # A school j is visited if flow leaves j
                        m_daily.addConstr(
                            v[j, w, t] == gp.quicksum(z[j, k, w, t] for k in nodes if k != j),
                            name=f"visit_to_out_arcs_j{j}_w{w}_t{t}"
                        )

            # Link day-on variable d_day[w,t] to total visits
            BIG = len(s_idx) if len(s_idx) > 0 else 1
            for w in W:
                for t in D:
                    # If any visit happens, the day is on
                    m_daily.addConstr(
                        gp.quicksum(v[j, w, t] for j in s_idx) >= d_day[w, t],
                        name=f"day_on_if_any_visit_w{w}_t{t}"
                    )
                    # If the day is on, visits are possible (BIG M)
                    m_daily.addConstr(
                        gp.quicksum(v[j, w, t] for j in s_idx) <= BIG * d_day[w, t],
                        name=f"visits_only_if_day_on_w{w}_t{t}"
                    )

            MAX_SCHOOLS_PER_DAY = 2
            for w in target_week:
                for t in D:
                    # Max 2 schools visited per working day
                    m_daily.addConstr(
                        gp.quicksum(v[j, w, t] for j in s_idx) <= MAX_SCHOOLS_PER_DAY * d_day[w, t],
                        name=f"cap_schools_per_day_w{w}_t{t}"
                    )

            # Daily Time Capacity Constraint
            for w in target_week:
                for t in D:
                    m_daily.addConstr(
                        gp.quicksum(x_d[i, w, t] * theta[i] for i in students) + # Session time
                        gp.quicksum(z[j, k, w, t] * time_matrix[j, k] # Travel time
                                     for j in nodes for k in nodes if j != k)
                        <= theta_max,
                        name=f"cap_time_w{w}_t{t}"
                    )

            m_daily.setObjective(
                gp.quicksum(x_d[i, w, t]
                            for i in students for w in target_week for t in D),
                GRB.MAXIMIZE
            )

            status_placeholder.write("Solving **Daily Subproblem**...")
            progress_bar.progress(min(progress + 10, 90))
            m_daily.optimize()

            if m_daily.status == GRB.OPTIMAL:
                status_placeholder.success(f"**Optimization Complete!** Feasible daily solution found (Weekly Cap={cap_total_value}, Max Schools/Week={max_schools_per_week_value}).")
                progress_bar.progress(100)

                # ---- Extract daily decision variables into a dictionary ----
                final_daily_var = {}
                for i in students:
                    for w in W:
                        for t in D:
                            var = m_daily.getVarByName(f"x[{i},{w},{t}]")
                            if var is not None:
                                final_daily_var[(i, w, t)] = var.X
                            else:
                                final_daily_var[(i, w, t)] = 0
                
                # --- FIX: Extract Routing Variables (z) into a dictionary immediately ---
                final_daily_z_var = {}
                for j in nodes:
                    for k in nodes:
                        if j != k:
                            for w in W:
                                for t in D:
                                    var = m_daily.getVarByName(f"z[{j},{k},{w},{t}]")
                                    if var is not None and var.X > 0.5:
                                        final_daily_z_var[(j, k, w, t)] = var.X

                # Store the results in session state
                st.session_state.final_daily = m_daily
                st.session_state.final_daily_var = final_daily_var
                st.session_state.final_daily_z_var = final_daily_z_var # STORE Z VARS
                st.session_state.optimization_success = True
                
                # --- CRITICAL FIX: Snapshot the current data context into session state ---
                st.session_state.saved_student_school = student_school.copy()
                st.session_state.saved_school_names = school_names.copy()
                st.session_state.saved_students = list(students)
                st.session_state.saved_theta = theta.copy()
                st.session_state.saved_p = p.copy()
                st.session_state.saved_I = I.copy()
                st.session_state.saved_assigned_schools_df = assigned_schools.copy()
                
                break


            if m_daily.status == GRB.INFEASIBLE:
                status_placeholder.warning("Daily subproblem infeasible! Tightening weekly caps and re-solving the master problem...")
                cap_total_value -= CAP_DECREASE
                max_schools_per_week_value -= SCHOOLS_DECREASE

                if cap_total_value < MIN_CAP_TOTAL or max_schools_per_week_value < MIN_SCHOOLS_PER_WEEK:
                    status_placeholder.error("Weekly caps became too tight. No feasible solution found under current constraints. Stopping.")
                    break

                for w in W:
                    cap_total_constraints[w].RHS = cap_total_value
                    max_schools_constraints[w].RHS = max_schools_per_week_value

                weekly.update()
                continue

            status_placeholder.error("Daily model status neither OPTIMAL nor INFEASIBLE. Stopping.")
            break

        if st.session_state.optimization_success:
            pass # Success message already displayed
        else:
            if 'weekly' in locals() and weekly.status == GRB.OPTIMAL:
                status_placeholder.error("No feasible daily VRP solution found for the calculated weekly pattern.")
            else:
                status_placeholder.error("No feasible weekly solution found.")

    except Exception as e:
        status_placeholder.error(f"Error during optimization: {e}")
        progress_bar.progress(0)

st.markdown("---")

# ---------------------------------------------------------
#        SIDEBAR — VISUALIZATION SETTINGS (AFTER OPTIMIZATION)
# ---------------------------------------------------------

st.sidebar.header("📊 Visualization Settings", divider="gray")

max_weeks_display = st.sidebar.number_input(
    "Maximum number of weeks to display in plots",
    min_value=1,
    max_value=len(W),
    value=16,
    step=1
)

st.sidebar.header("Detailed Visualization Inputs", divider="gray")

# --- Helper to Organize Students by School ---
students_by_school = {}
for i in students:
    # student_school[i] is 1-based index from model. Convert to 0-based.
    s_idx_1based = student_school[i]
    s_idx_0based = s_idx_1based - 1
    
    # Access School Name from the assigned_schools DataFrame
    if 0 <= s_idx_0based < len(assigned_schools):
        s_name = assigned_schools.iloc[s_idx_0based]['School_Name']
        if s_name not in students_by_school:
            students_by_school[s_name] = []
        students_by_school[s_name].append(i)

# --- 2-Step Dropdown for Student Selection ---
school_options = sorted(list(students_by_school.keys()))

if school_options:
    selected_viz_school = st.sidebar.selectbox("Filter by School", school_options)
    
    # Get students for the selected school
    student_options = students_by_school[selected_viz_school]
    
    # Select Student ID from the filtered list
    student_id_input = st.sidebar.selectbox("Select Student ID", student_options)
else:
    st.sidebar.warning("No students generated.")
    student_id_input = 0
    
default_student_id = student_id_input

month_input = st.sidebar.number_input(
    "Enter Month for to view the monthly calender of the clinician",
    min_value=1,
    max_value=12,
    value=1,
    step=1
)

week_input = st.sidebar.number_input(
    "Enter Week for Itinerary/Route Plots",
    min_value=1,
    max_value=199,
    value=1,
    step=1
)

# ============================================================
# OPTIMIZATION OUTPUT VISUALIZATION SECTION (BOTTOM)
# ============================================================

## Optimization Output Visualization 📊
st.header("Optimal Schedule Visualizations", divider="gray")

if st.session_state.optimization_ran and st.session_state.optimization_success:
    
    # --- Retrieve data from session state ---
    x_li_weekly = st.session_state.x_li_weekly
    aux1_weekly = st.session_state.aux1_weekly
    final_daily = st.session_state.final_daily
    final_daily_var = st.session_state.final_daily_var
    final_daily_z_var = st.session_state.final_daily_z_var # Retrieve Z vars

    # --- Retrieve SNAPSHOTTED Context Data ---
    # We must use these saved versions for visualization to ensure indices match the variables
    ctx_student_school = st.session_state.saved_student_school
    ctx_school_names = st.session_state.saved_school_names
    ctx_students = st.session_state.saved_students
    ctx_theta = st.session_state.saved_theta
    ctx_p = st.session_state.saved_p
    ctx_I = st.session_state.saved_I
    ctx_assigned_schools = st.session_state.saved_assigned_schools_df
    
    # --- Proceed with visualization ---
    weeks_plot = [w for w in sorted(W) if 1 <= w <= max_weeks_display]

    st.subheader("Weekly Appointments by Student Category")
    st.markdown(
    """
    This stacked bar chart illustrates the weekly workload distribution across the four student categories. **Note:** High Intensity (HI) sessions are weighted by 3 in this chart to reflect their higher frequency (e.g., a student starting a HI block consumes 3 weekly capacity slots).
    """)

    weekly_counts = {}
    for w in W:
        weekly_counts[(w, 'Low', 1)] = 0 # LI-HP
        weekly_counts[(w, 'Low', 2)] = 0 # LI-LP
        weekly_counts[(w, 'High', 1)] = 0 # HI-HP (Weighted by 3)
        weekly_counts[(w, 'High', 2)] = 0 # HI-LP (Weighted by 3)

    for i in ctx_students:
        for w in W:
            if ctx_I[i] == 0: # Low Intensity
                weekly_counts[(w, 'Low', ctx_p[i])] += x_li_weekly.get((i, w), 0)
            else: # High Intensity
                weekly_counts[(w, 'High', ctx_p[i])] += aux1_weekly.get((i, w), 0)

    # Low Priority (p=2) - LP
    low_int_low_pri = [weekly_counts.get((w, 'Low', 2), 0) for w in weeks_plot] # LI-LP (Green)
    high_int_low_pri = [weekly_counts.get((w, 'High', 2), 0) for w in weeks_plot] # HI-LP (Orange)
    
    # High Priority (p=1) - HP
    low_int_high_pri = [weekly_counts.get((w, 'Low', 1), 0) for w in weeks_plot] # LI-HP (Blue)
    high_int_high_pri = [weekly_counts.get((w, 'High', 1), 0) for w in weeks_plot] # HI-HP (Red)


    bar_width = 0.8
    x = np.arange(len(weeks_plot))

    fig, ax = plt.subplots(figsize=(12, 5))
    
    # LI-LP (Green)
    ax.bar(x, low_int_low_pri, color=COLOR_LI_LP, width=bar_width, label="LI-LP (Low Intensity, Low Priority)")
    
    # LI-HP (Blue) on top of LI-LP
    bottom_li_lp = np.array(low_int_low_pri)
    ax.bar(x, low_int_high_pri, bottom=bottom_li_lp, color=COLOR_LI_HP, width=bar_width, label="LI-HP (Low Intensity, High Priority)")
    
    bottom_li_hp = bottom_li_lp + np.array(low_int_high_pri) # Total LI bottom (Base for HI)

    # HI-LP (Orange) on top of LI total
    bottom_hi_lp = bottom_li_hp # Total LI is the base for HI-LP
    ax.bar(x, high_int_low_pri, bottom=bottom_hi_lp, color=COLOR_HI_LP, width=bar_width, label="HI-LP (High Intensity, Low Priority) [×3 Weighted]")

    # HI-HP (Red) on top of LI total and HI-LP
    bottom_hi_hp = bottom_hi_lp + np.array(high_int_low_pri)
    ax.bar(x, high_int_high_pri, bottom=bottom_hi_hp, color=COLOR_HI_HP, width=bar_width, label="HI-HP (High Intensity, High Priority) [×3 Weighted]")

    ax.axhline(cap_total, color='red', linestyle='--', label='Weekly Capacity Limit', alpha=0.7)

    ax.set_xlabel("Week", fontsize=11)
    ax.set_ylabel("Weighted Weekly Appointments (Capacity Slots)", fontsize=11)
    ax.set_title(f"Weekly Appointment Load by Student Type (Weeks 1-{max_weeks_display})", fontsize=14)
    ax.set_xticks(x[::5])
    ax.set_xticklabels(weeks_plot[::5], rotation=45)
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Disaggregated Weekly Appointments")
    st.markdown(
    """
    These side-by-side bar charts clearly isolate the relative impact of **Intensity** and **Priority** on the weekly schedule. """)

    col_intensity, col_priority = st.columns(2)

    with col_intensity:
        weekly_low_int = []
        weekly_high_int = []
        for w in weeks_plot:
            low_weighted = sum(x_li_weekly.get((i, w), 0) for i in ctx_students if ctx_I[i] == 0)
            high_weighted = sum(aux1_weekly.get((i, w), 0) for i in ctx_students if ctx_I[i] == 1)
            weekly_low_int.append(low_weighted)
            weekly_high_int.append(high_weighted)

        x2 = np.arange(len(weeks_plot))
        width2 = 0.35

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.bar(x2 - width2 / 2, weekly_low_int, width2, label="Low Intensity (x1)", color="#4fa3ff")
        ax2.bar(x2 + width2 / 2, weekly_high_int, width2, label="High Intensity (x3)", color="#ff6a6a")

        ax2.set_xlabel("Week")
        ax2.set_ylabel("Weighted Appointments")
        ax2.set_title("Weekly Appointments by Intensity")
        ax2.set_xticks(x2[::5])
        ax2.set_xticklabels(weeks_plot[::5], rotation=45)
        ax2.legend(fontsize=8)
        ax2.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)

    with col_priority:
        weekly_pri1 = []
        weekly_pri2 = []

        for w in weeks_plot:
            pri1_weighted = 0
            pri2_weighted = 0
            for i in ctx_students:
                # Add weighted sessions (1 for LI, 3 for HI)
                weight = 1 if ctx_I[i] == 0 else 3 
                sessions = x_li_weekly.get((i, w), 0) if ctx_I[i] == 0 else aux1_weekly.get((i, w), 0)
                
                if ctx_p[i] == 1: # High Priority
                    pri1_weighted += sessions
                else: # Low Priority
                    pri2_weighted += sessions
            weekly_pri1.append(pri1_weighted)
            weekly_pri2.append(pri2_weighted)

        x3 = np.arange(len(weeks_plot))
        width3 = 0.35

        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.bar(x3 - width3 / 2, weekly_pri1, width3, label="Priority 1 (High)", color="#34c759")
        ax3.bar(x3 + width3 / 2, weekly_pri2, width3, label="Priority 2 (Low)", color="#ff9500")

        ax3.set_xlabel("Week")
        ax3.set_ylabel("Weighted Appointments")
        ax3.set_title("Weekly Appointments by Priority")
        ax3.set_xticks(x3[::5])
        ax3.set_xticklabels(weeks_plot[::5], rotation=45)
        ax3.legend(fontsize=8)
        ax3.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig3)

    st.markdown("---")

    # ============================================================
    # EXPANDERS FOR NEW VISUALIZATIONS
    # ============================================================

    # ---------- 1. Student Schedule Plot ----------
    with st.expander("🔬 Detailed Student Schedule Analysis"):
        st.markdown(
            """
            Analyze an individual student's schedule over the planning horizon. On the left, enter the school name, then pick one student.
            """
        )
        col_sch, col_cal = st.columns(2)
        
        # Plot 1: Weekly Session Block
        with col_sch:
            def plot_student_schedule(student_id, week_limit=50):
                if student_id not in ctx_students:
                    st.write("Student ID out of range or student's school not selected for optimization.")
                    return
                
                if ctx_I[student_id] == 0:
                    session_weeks = [w for w in W if x_li_weekly.get((student_id, w), 0) > 0]
                else:
                    session_weeks = [w for w in W if aux1_weekly.get((student_id, w), 0) > 0]

                intensity = "High" if ctx_I[student_id] == 1 else "Low"
                priority = "High" if ctx_p[student_id] == 1 else "Low"

                fig_s, ax_s = plt.subplots(figsize=(6, 2))
                
                # FIX: Removed marker and markersize which caused the AttributeError
                ax_s.eventplot(session_weeks,
                                 colors=COLOR_LI_HP if intensity == "Low" else COLOR_HI_HP,
                                 lineoffsets=0.5,
                                 linelengths=0.8,
                                 orientation='horizontal')
                                
                ax_s.set_title(f"Student {student_id}: Treatment Block (Weeks 1-{week_limit})", fontsize=10)
                ax_s.set_xlabel("Week")
                ax_s.set_yticks([])
                ax_s.set_xlim(1, week_limit)
                ax_s.grid(True, axis='x', linestyle='--', alpha=0.6)
                plt.tight_layout()
                st.pyplot(fig_s)
                
                st.markdown(f"**Details:** Priority: **{priority}**, Intensity: **{intensity}**, Duration: **{ctx_theta[student_id]} min/session**")


            student_id_int = int(default_student_id)
            if student_id_int in ctx_students:
                plot_student_schedule(student_id_int, week_limit=max_weeks_display)
            else:
                st.warning(f"Student ID **{student_id_int}** is not in the optimized set.")

        # Plot 2: Daily Session Calendar
        with col_cal:
            def plot_student_daily_calendar(student_id):
                if student_id not in ctx_students:
                    st.write("Student ID out of range or student's school not selected for optimization.")
                    return

                student_days = [
                    (w, d)
                    for (i, w, d), val in final_daily_var.items()
                    if i == student_id and val > 0.5 and w <= max_weeks_display
                ]

                if not student_days:
                    st.info("No daily sessions found for this student.")
                    return

                fig, ax = plt.subplots(figsize=(6, 2.5))

                # Map model day index (1 to num_days) to real day index (1 to 5)
                model_day_to_real_day = {i+1: wd for i, wd in enumerate(working_weekdays)}

                xs = [w for (w, d) in student_days]
                ys = [model_day_to_real_day[d] for (w, d) in student_days]

                intensity_color = COLOR_HI_HP if ctx_I[student_id] == 1 else COLOR_LI_HP

                ax.scatter(xs, ys, color=intensity_color, s=40, alpha=0.8)

                weekday_labels = [weekday_names_map[wd] for wd in working_weekdays]
                
                ax.set_yticks(working_weekdays) 
                ax.set_yticklabels(weekday_labels)
                ax.set_xlabel("Week")
                ax.set_ylabel("Day of Week")
                ax.set_title(f"Student {student_id}: Daily Schedule (Weeks 1-{max_weeks_display})", fontsize=10)
                if working_weekdays:
                     ax.set_ylim(min(working_weekdays)-0.5, max(working_weekdays)+0.5)

                ax.grid(True, linestyle="--", alpha=0.5)
                plt.tight_layout()
                st.pyplot(fig)

            if student_id_int in ctx_students:
                plot_student_daily_calendar(student_id_int)
            else:
                pass

    st.markdown("---")

    # ---------- 3. Monthly Calendar View ----------
    with st.expander("📅 Monthly Calendar View"):
        st.markdown(
            """
            This calendar view provides a clear, at-a-glance visualization of the clinician's activities for the selected month. Days with scheduled sessions are highlighted in green, showing the schools visited and the total number of High Intensity (HI) and Low Intensity (LI) student sessions planned.
            """
        )
        def build_date_mapping(W, D, working_weekdays, anchor_date):
            days_since_monday = anchor_date.weekday()
            first_monday = anchor_date - datetime.timedelta(days=days_since_monday)
            
            mapping = {}
            for w in W:
                monday_of_current_week = first_monday + datetime.timedelta(weeks=(w-1))
                for t in D:
                    real_wd_idx = working_weekdays[t-1] 
                    day_offset = real_wd_idx - 1 
                    actual_date = monday_of_current_week + datetime.timedelta(days=day_offset)
                    mapping[(w, t)] = actual_date
            return mapping

        date_map = build_date_mapping(W, D, working_weekdays, ANCHOR_DATE)

        def plot_month_calendar(year, month):
            day_info = {}
            
            # USE SAVED ASSIGNED SCHOOLS FOR INDEX LOOKUP
            local_school_idx_to_name = { 
                idx + 1: ctx_assigned_schools.loc[idx, "School_Name"] 
                for idx in ctx_assigned_schools.index
            }

            for (w, t), date in date_map.items():
                if date.year != year or date.month != month:
                    continue

                schools_today = set()
                HI_count = 0
                LI_count = 0
                
                # Ensure all lookup keys are explicitly cast to standard Python int for robustness
                _w = int(w)
                _t = int(t)

                for i in ctx_students:
                    # FIX: Use final_daily_var dictionary and explicitly cast student ID key
                    _i = int(i)
                    val = final_daily_var.get((_i, _w, _t), 0.0)
                    if val > 0.5:
                        if ctx_I[_i] == 1:
                            HI_count += 1
                        else:
                            LI_count += 1
                        
                        j = ctx_student_school[_i]
                        schools_today.add(str(local_school_idx_to_name.get(j, f"S_Err_{j}")))


                if HI_count + LI_count > 0:
                    day_info[date.day] = {
                        "schools": sorted(list(schools_today)),
                        "HI": HI_count,  
                        "LI": LI_count
                    }

            cal = calendar.Calendar(firstweekday=0)  # Monday
            month_grid = cal.monthdayscalendar(year, month)

            fig_c, ax_c = plt.subplots(figsize=(12, 8))
            ax_c.set_axis_off()

            day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            for col, name in enumerate(day_names):
                ax_c.text(col + 0.5, 1, name, fontsize=14, ha="center", va="bottom", weight='bold')

            for row, week in enumerate(month_grid):
                for col, day in enumerate(week):
                    x0 = col
                    y0 = -row - 1

                    if day != 0:
                        date_obj = datetime.date(year, month, day)
                        weekday = date_obj.weekday() # 0=Mon, 6=Sun
                        
                        is_working_day = (weekday + 1) in working_weekdays
                        
                        if not is_working_day:
                             rect_color = "#e0e0e0" # Light Grey for non-working
                             text_color = "grey"
                        else:
                             rect_color = "white" 
                             text_color = "black"

                        if day in day_info: # Day with scheduled sessions
                             rect_color = "#90ee90" # Light green
                             text_color = "black"

                        ax_c.add_patch(plt.Rectangle((x0, y0), 1, 1, facecolor=rect_color, edgecolor="black", zorder=0, alpha=0.9))
                        
                        ax_c.text(x0 + 0.05, y0 + 0.9, str(day), fontsize=12, weight="bold", color=text_color)

                        if day in day_info:
                            info = day_info[day]
                            schools = info["schools"]
                            HI_count = info["HI"]
                            LI_count = info["LI"]

                            school_str = ", ".join(schools) if schools else "N/A"
                            if len(school_str) > 20: school_str = school_str[:18] + "..."
                            
                            # Line 1: Schools visited
                            ax_c.text(x0 + 0.5, y0 + 0.60, f"**{school_str}**", 
                                    fontsize=9, ha="center", va="center", weight='bold')

                            # Line 2: HI count (Red) and LI count (Blue/Green)
                            ax_c.text(x0 + 0.5, y0 + 0.35, f"HI:{HI_count} LI:{LI_count}", 
                                    fontsize=9, ha="center", va="center")
                                    
                            # Note: Achieving two different colors (red/green) in a single text object in Matplotlib is complex
                            # and usually requires using annotations or Tex/math text. Sticking to simple text is safer.
                            # For simple color distinction, we'll rely on the visual separation of the numbers.

            ax_c.set_title(f"{calendar.month_name[month]} {year} – Clinician Optimal Daily Schedule", fontsize=18, weight='bold')
            ax_c.set_xlim(0, 7)
            ax_c.set_ylim(-len(month_grid) - 1, 1.5)
            plt.tight_layout()
            st.pyplot(fig_c)

        plot_month_calendar(PLANNING_YEAR, int(month_input))
    
    st.markdown("---")

    # ---------- 4. Travels Per Day (given week) - RESTORED PLOT ----------
    with st.expander("📊 Travels Per Day in a Given Week"):
        st.write("The number of travel arcs (trips between locations, including to/from the office) required for each working day in the selected week. (Note that including the trips from and to the office, the travels must be at most 3 per day, which means at most 2 schools per day)")
        
        def plot_travels_per_day(week):
            day_labels = []
            travel_counts = []
            # Ensure week is explicitly cast
            _week = int(week)
            for t in sorted(D):
                daily_travel = 0.0
                # Ensure day is explicitly cast
                _day = int(t)

                for j in nodes:
                    for k in nodes:
                        if j == k:
                            continue
                        
                        # FIX: Use the extracted Z dictionary and explicit keys
                        val = st.session_state.final_daily_z_var.get((j, k, _week, _day), 0.0)
                        
                        if val > 0.5: # Use > 0.5 to count binary flows
                            daily_travel += val
                
                # Map t to real day name
                real_wd_idx = working_weekdays[_day-1]
                day_name = weekday_names_map.get(real_wd_idx)
                day_labels.append(f"{day_name} (Day {_day})")
                travel_counts.append(daily_travel)

            fig_t, ax_t = plt.subplots(figsize=(8, 5))
            ax_t.bar(day_labels, travel_counts, color="#6c5ce7")
            ax_t.set_title(f"Number of Travels per Day – Week {_week}")
            ax_t.set_xlabel("Day")
            ax_t.set_ylabel("Number of Travels (Arcs)")
            ax_t.grid(axis="y", alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig_t)

        plot_travels_per_day(int(week_input))

    st.markdown("---")

    # ---------- 5. Therapy vs Travel Time Ratio ----------
    with st.expander("⏱️ Weekly Time Allocation: Therapy vs. Travel"):
        st.markdown(
            """
            This chart visualizes the efficiency of the weekly schedule by showing the ratio of **Therapy Time** to **Total Time (Therapy + Travel)**. A ratio closer to $1.0$ indicates higher efficiency (more time spent on treatment, less on the road).
            """
        )
        therapy_time = {w: 0.0 for w in W}
        travel_time = {w: 0.0 for w in W}

        for w in W:
            _w = int(w)
            for t in D:
                _t = int(t)
                # Calculate therapy time
                for i in ctx_students:
                    _i = int(i)
                    # FIX: Use final_daily_var dictionary and explicitly cast student ID key
                    val = final_daily_var.get((_i, _w, _t), 0.0)
                    if val > 0.5:
                        therapy_time[_w] += ctx_theta[_i]
                # Calculate travel time
                for j in nodes:
                    for k in nodes:
                        if j != k:
                            # FIX: Use extracted Z dictionary and explicit keys
                            val = final_daily_z_var.get((j, k, _w, _t), 0.0)
                            if val > 0.5:
                                travel_time[_w] += time_matrix[j, k]

        ratio = {}
        for w in W:
            T = therapy_time[w]
            R = travel_time[w]
            if T + R > 0:
                ratio[w] = T / (T + R)
            else:
                ratio[w] = np.nan

        weeks_sorted = [w for w in W if w <= max_weeks_display]
        ratio_list = [ratio.get(w, np.nan) for w in weeks_sorted]
        
        valid_weeks = [w for w, r_val in zip(weeks_sorted, ratio_list) if not np.isnan(r_val)]
        valid_ratios = [r_val for r_val in ratio_list if not np.isnan(r_val)]


        fig_r, ax_r = plt.subplots(figsize=(10, 4))
        if valid_weeks:
            ax_r.plot(valid_weeks, valid_ratios, marker="o", color=COLOR_ROUTE, linewidth=2, markersize=5)
            ax_r.set_xlim(valid_weeks[0], valid_weeks[-1]) 
            
        ax_r.set_xlabel("Week")
        ax_r.set_ylabel("Efficiency Ratio (Therapy Time / Total Time)")
        ax_r.set_title(f"Weekly Schedule Efficiency (Weeks 1-{max_weeks_display})")
        ax_r.set_ylim(0, 1.05)
        ax_r.grid(True)
        plt.tight_layout()
        st.pyplot(fig_r)

    st.markdown("---")

    # ---------- 6. Daily Route Network (Itinerary + Map) ----------
    with st.expander("📍 Daily Itinerary and Optimized Route Map"):
        st.markdown(
            """
            This is the most detailed part of the schedule, providing the step-by-step route for each working day of the selected week.
            """
        )
        
        # Modified to accept z_dict instead of model object
        @st.cache_data(show_spinner=False, max_entries=50)
        def reconstruct_path_for_day(week, day, _z_dict, nodes, time_matrix_data, num_schools):
            arcs = set()
            _week = int(week)
            _day = int(day)

            for j in nodes:
                for k in nodes:
                    if j == k:
                        continue
                    # FIX: Read from the extracted dictionary, not the Gurobi model object
                    val = _z_dict.get((j, k, _week, _day), 0.0)
                    if val > 0.5:
                        arcs.add((j, k))
            
            if not arcs:
                return None, None
            
            path = []
            current = 0 # Start at depot
            
            while True: 
                
                # Find the next node (k) originating from the current node (j)
                nxt_info = next(((k, time_matrix_data[current, k]) for (j, k) in arcs if j == current), None)
                
                if nxt_info is None:
                    # If we can't find an outgoing arc, we stop. Should be at depot (0).
                    break
                
                nxt, travel_time = nxt_info
                
                path.append((current, travel_time))

                arcs.discard((current, nxt))
                current = nxt
            
            # The path must start at 0 and end at 0. If current != 0, it means the last arc to 0 wasn't explicitly found
            # but for a feasible VRP solution, it must return. 
            if current != 0 and path and len(path) > 1:
                    # Add the final travel arc back to depot
                    last_stop = path[-1][0]
                    final_travel_time = time_matrix_data[last_stop, 0]
                    path.append((0, final_travel_time)) 

            nodes_path = [p[0] for p in path]
            # Travel times are for the arcs *between* the nodes. 
            travel_times = [p[1] for p in path] 
            
            return nodes_path, travel_times 

        def get_sessions_at_school(j, week, day):
            sessions = []
            
            # USE SNAPSHOTTED MAPPING (ctx_student_school) to ensure indices match the variables
            students_at_school = [i for i, school_j in ctx_student_school.items() if school_j == j]
            
            # Ensure all lookup keys are explicitly cast to standard Python int for robustness
            _week = int(week)
            _day = int(day)
            
            for i in students_at_school:
                _i = int(i)
                # FIX: Use the explicit int-casted keys for retrieval
                val = final_daily_var.get((_i, _week, _day), 0.0)
                
                if val > 0.5:
                    intensity = "HI" if ctx_I[_i] == 1 else "LI"
                    priority = "HP" if ctx_p[_i] == 1 else "LP"
                    sessions.append({
                        "id": _i,
                        "type": f"{intensity}-{priority}",
                        "duration": ctx_theta[_i],
                        "color": COLOR_HI_HP if intensity == "HI" else (COLOR_LI_HP if priority == "HP" else COLOR_LI_LP)
                    })
            return sessions


        def print_day_itinerary(week, day):
            
            # --- FIX: HARD CHECK for Zero Therapy Days to avoid "ghost routes" ---
            total_sessions_count = 0
            _week = int(week)
            _day = int(day)
            
            for i in ctx_students:
                if final_daily_var.get((int(i), _week, _day), 0.0) > 0.5:
                    total_sessions_count += 1
            
            # Map t (day) to real weekday
            real_wd_idx = working_weekdays[_day-1]
            day_name = weekday_names_map.get(real_wd_idx)
            
            st.markdown(f"### 🗓️ Week {week}, **{day_name}** (Day {day}) Itinerary")

            if total_sessions_count == 0:
                st.info("No appointments scheduled for this day.")
                return
            
            # ---------------------------------------------------------------------

            
            # FIX: Ensure we use the SAVED list of nodes/schools for visualization consistency
            saved_num_schools = len(ctx_assigned_schools)
            viz_nodes = range(saved_num_schools + 1)

            # 1. Get the actual route path and travel times using the extracted dictionary
            nodes_path, travel_times = reconstruct_path_for_day(
                week, day, final_daily_z_var, viz_nodes, time_matrix, saved_num_schools
            )
            

            if not nodes_path or len(nodes_path) <= 1:
                st.info("No visits scheduled for this day.")
                return

            itinerary = []
            processed_schools = set()
            total_travel_time = 0.0
            total_therapy_time = 0.0
            
            # Start: path[0] must be Depot (0)
            itinerary.append(f"**Start**: Leave **{ctx_school_names.get(nodes_path[0], 'Depot')}**")
            
            # Iterate through the route
            for i in range(len(nodes_path) - 1):
                j = nodes_path[i] # Current location
                k = nodes_path[i+1] # Next stop
                travel_time_min = travel_times[i+1] if i+1 < len(travel_times) else 0 # travel time from j to k is stored at index i+1
                
                # Travel step
                if j != k:
                    j_name = ctx_school_names.get(j, 'Depot')
                    k_name = ctx_school_names.get(k, 'Depot')
                    total_travel_time += travel_time_min
                    
                    if k != 0:
                        itinerary.append(f"**Travel**: **{j_name}** → **{k_name}** ({travel_time_min:.1f} min)")
                    else: # Return to Depot
                             itinerary.append(f"**End**: Return to **{k_name}** ({travel_time_min:.1f} min travel)")

                # Activity at next node k
                if k != 0 and k not in processed_schools:
                    sessions = get_sessions_at_school(k, week, day)
                    if sessions:
                        total_duration = sum(s['duration'] for s in sessions)
                        total_therapy_time += total_duration
                        # Use Saved DF for ID lookup
                        school_id = ctx_assigned_schools.loc[k-1, 'School_ID']
                        # Use HTML for color styling here, as it's in st.markdown
                        itinerary.append(f"  - **Arrive at {ctx_school_names.get(k, 'School Error')} (ID: {school_id})**: {len(sessions)} session(s) scheduled (Therapy time: **{total_duration:.0f} min**).")
                        
                        for s in sessions:
                            itinerary.append(f"  - Session: Student **{s['id']}** (<span style='color:{s['color']}; font-weight:bold;'>{s['type']}</span>, {s['duration']} min)")
                        
                        processed_schools.add(k)
            
            st.markdown(f"**Total Daily Time Allocation:** Therapy: **{total_therapy_time:.0f} min** | Travel: **{total_travel_time:.1f} min**")
            st.markdown("\n".join(itinerary), unsafe_allow_html=True)
            
            st.markdown("#### 🗺️ Daily Route Map")
            
            # --- Map Visualization ---
            route_coords = {}
            for node_idx in set(nodes_path):
                if node_idx == 0:
                    route_coords[0] = depot_coords
                else:
                    school_df_index = node_idx - 1
                    if 0 <= school_df_index < len(ctx_assigned_schools):
                        school_row = ctx_assigned_schools.iloc[school_df_index]
                        route_coords[node_idx] = school_row["Coordinates"]
            
            daily_fmap = folium.Map(location=[depot_lat, depot_lon], zoom_start=11, tiles="OpenStreetMap")
            
            # Draw Route Lines
            path_coords = [route_coords[node] for node in nodes_path if node in route_coords and route_coords[node] is not None]
            if path_coords:
                folium.PolyLine(path_coords, color=COLOR_ROUTE, weight=4, opacity=0.8).add_to(daily_fmap)
            
            # Mark Depot
            folium.Marker(
                depot_coords,
                icon=folium.Icon(color='green', icon='briefcase', prefix='fa'),
                popup="**Depot (Centroid)**"
            ).add_to(daily_fmap)

            # Mark Schools in Order
            unique_stops = [node for node in nodes_path if node != 0]
            
            for idx, node_idx in enumerate(unique_stops, start=1):
                if node_idx in route_coords and route_coords[node_idx] is not None:
                    lat, lon = route_coords[node_idx]
                    sname = ctx_assigned_schools.loc[node_idx - 1, 'School_Name']
                    
                    # USE UNIFORM COLOR FOR ALL SCHOOLS IN ITERATIVE MAP
                    color = COLOR_SCHOOL_UNIFORM
                    
                    # School Marker (Circle)
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=8,
                        color=color, # Use the uniform color here
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.7,
                        popup=f"Stop **#{idx}**<br>School: **{sname}**"
                    ).add_to(daily_fmap)
                    
                    # Add stop number label (e.g., #1)
                    folium.Marker(
                        [lat, lon],
                        icon=folium.DivIcon(
                            html=f'<div style="font-size: 14pt; color: black; font-weight: bold; transform: translate(-50%, -150%);">#{idx}</div>'
                        )
                    ).add_to(daily_fmap)
                    
                    # Add school name label
                    folium.Marker(
                        [lat, lon],
                        icon=folium.DivIcon(
                            html=f'<div style="font-size: 10pt; color: {color}; font-weight: bold; transform: translate(15px, -10px);">{sname}</div>'
                        )
                    ).add_to(daily_fmap)


            st_folium(daily_fmap, width=None, height=450, key=f"map_{week}_{day}")
            
        
        for t in sorted(D):
            print_day_itinerary(int(week_input), t)

else:
    st.info("Run the optimization to see the output visualizations.")