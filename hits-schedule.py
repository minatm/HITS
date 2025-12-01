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

# ---------------------------------------------------------
#     PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Clinician Tool", layout="wide")

# --- Initialize Session State ---
if 'optimization_ran' not in st.session_state:
    st.session_state.optimization_ran = False
if 'optimization_success' not in st.session_state:
    st.session_state.optimization_success = False
if 'final_daily' not in st.session_state:
    st.session_state.final_daily = None
if 'final_daily_var' not in st.session_state:
    st.session_state.final_daily_var = {}
if 'x_li_weekly' not in st.session_state:
    st.session_state.x_li_weekly = {}
if 'aux1_weekly' not in st.session_state:
    st.session_state.aux1_weekly = {}
if 'tr_weekly' not in st.session_state:
    st.session_state.tr_weekly = {}

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
    </style>
    """,
    unsafe_allow_html=True
)
st.title("HITS: Clinician Scheduling Tool")
st.markdown(
    """
This dashboard provides the optimal schedules for clinicians participating in both Normal and HITS SLP sessions. It allows you to enter clinician information, school demand, the number of students in each priority and intensity category, and each student’s referral date. You can also specify the number of required sessions, the planning horizon, and the priority rule assigned to high priority students.

In the first step, upload the clinician data files. These should include each clinician’s assigned schools. Then upload the student data files, which should include the geographical locations of the schools.
    """
)

# ---------------------------------------------------------
#     SIDEBAR — FILE UPLOADS FIRST
# ---------------------------------------------------------
st.sidebar.header("Upload the Clinician and School data files (format: json)", divider="gray")

slp_file = st.sidebar.file_uploader("Upload Clinician data file (slp.json)", type="json")
school_file = st.sidebar.file_uploader("Upload School data file (schools.json)", type="json")

if not slp_file or not school_file:
    st.warning("Please upload both JSON files to continue.")
    st.stop()

slp_df = pd.read_json(slp_file)
school_df = pd.read_json(school_file)

# Global weeks horizon (for model)
W = list(range(1, 200))  # 1..199

# --- CORE FIX: RENAME 'School' COLUMN TO 'School_Name' FOR CONSISTENCY ---
if "School" in school_df.columns:
    school_df.rename(columns={"School": "School_Name"}, inplace=True)
elif "School_Name" not in school_df.columns:
     # Fallback if 'School' or 'School_Name' is missing
    school_df["School_Name"] = "School " + school_df["School_ID"].astype(str)
# -------------------------------------------------------------------------


# ---------------------------------------------------------
#     AUTO-DETECT POSTAL COLUMN
# ---------------------------------------------------------
possible_postal_cols = [
    "Postal_Code", "PostalCode", "postal_code",
    "Postal Code", "ZIP", "Zip", "PostCode"
]
postal_col = next((c for c in school_df.columns if c in possible_postal_cols), None)

if postal_col is None:
    st.error("Postal code column missing.")
    st.stop()

# ---------------------------------------------------------
#     SIDEBAR — CLINICIAN ID (MODIFIED with on_change)
# ---------------------------------------------------------
st.sidebar.header("Clinician Selection", divider="gray")

clinician_id_input = st.sidebar.number_input(
    "Enter Clinician ID to see the optimal schedule",
    min_value=0,
    step=1,
    on_change=reset_optimization_state  
)

# ---------------------------------------------------------
#     SIDEBAR — DEMAND MATRIX (EDITABLE TABLE)
# ---------------------------------------------------------
# ---------------------------------------------
# SIDEBAR — DEMAND MATRIX (EDITABLE TABLE)
# ---------------------------------------------
st.sidebar.header("Enter the number of students currently in the waitlist", divider="gray")

# Filter using the SAME logic that works on the right side
assigned_schools = school_df[
    school_df["SLP_IDs"].apply(
        lambda ids: clinician_id_input in ids if isinstance(ids, list) else False
    )
].copy()

# Initialize Demand if missing
if "Demand" not in assigned_schools.columns:
    np.random.seed(42)
    assigned_schools["Demand"] = np.random.randint(4, 10, size=len(assigned_schools))

# Only show the assigned schools in the editor, showing School_Name
demand_matrix = assigned_schools[["School_Name", "Demand"]]  

edited_matrix = st.sidebar.data_editor(
    demand_matrix,
    num_rows="fixed",
    use_container_width=True,
    height=300
)

# ---------------------------------------------------------
#     CREATE PLACEHOLDER FOR r EDITOR (will show after demand)
# ---------------------------------------------------------


# Write updated demand back to original dataframe
# Need to merge the updated demand back using the School_Name as a key if School_Name is unique
school_names_in_df = assigned_schools["School_Name"].tolist()
updated_demand_series = edited_matrix.set_index("School_Name")["Demand"]

for school_name, demand_val in updated_demand_series.items():
    school_df.loc[
        (school_df["School_Name"] == school_name) & 
        (school_df["SLP_IDs"].apply(lambda ids: clinician_id_input in ids if isinstance(ids, list) else False)),
        "Demand"
    ] = demand_val

# ---------------------------------------------------------
#     AUTO-POPULATE DEFAULT LI/HI/LP/HP COUNTS BASED ON DEMAND
# ---------------------------------------------------------

default_ratios = {
    "LI_LP": 0.40,
    "LI_HP": 0.20,
    "HI_LP": 0.20,
    "HI_HP": 0.20
}

# We must recompute assigned_schools after updating demand
assigned_schools = school_df[
    school_df["SLP_IDs"].apply(
        lambda ids: clinician_id_input in ids if isinstance(ids, list) else False
    )
].copy()

# Ensure columns exist
for col in ["LI_LP", "LI_HP", "HI_LP", "HI_HP"]:
    if col not in assigned_schools.columns:
        assigned_schools[col] = 0

# Fill defaults ONLY if totals don't match Demand
for idx, row in assigned_schools.iterrows():

    demand = int(row["Demand"])

    # If user already filled values correctly, skip
    existing_sum = row["LI_LP"] + row["LI_HP"] + row["HI_LP"] + row["HI_HP"]
    if existing_sum == demand and existing_sum > 0:
        continue

    # Proportional allocation
    li_lp = int(demand * default_ratios["LI_LP"])
    li_hp = int(demand * default_ratios["LI_HP"])
    hi_lp = int(demand * default_ratios["HI_LP"])
    hi_hp = int(demand * default_ratios["HI_HP"])

    # Fix rounding so totals match demand
    diff = demand - (li_lp + li_hp + hi_lp + hi_hp)
    li_lp += diff  # Add difference to one category

    # Store back
    assigned_schools.loc[idx, ["LI_LP", "LI_HP", "HI_LP", "HI_HP"]] = [
        li_lp, li_hp, hi_lp, hi_hp
    ]

# Write defaults to the true dataframe so editor sees them
for _, row in assigned_schools.iterrows():
    sname = row["School_Name"]
    school_df.loc[
        school_df["School_Name"] == sname, 
        ["LI_LP", "LI_HP", "HI_LP", "HI_HP"]
    ] = (row["LI_LP"], row["LI_HP"], row["HI_LP"], row["HI_HP"])


####adding for r
# ---------------------------------------------------------
#     SIDEBAR — r MATRIX EDITOR
# ---------------------------------------------------------



#####




# ---------------------------------------------------------
#     SIDEBAR — STUDENT TYPE / SESSION PARAMETERS
# ---------------------------------------------------------

st.sidebar.header("Enter Student Type Counts", divider="gray")

# Create default columns if missing
for col in ["LI_LP", "LI_HP", "HI_LP", "HI_HP"]:
    if col not in assigned_schools.columns:
        assigned_schools[col] = 0

# Editable table only for assigned schools, showing School_Name
type_editor = st.sidebar.data_editor(
    assigned_schools[["School_Name", "LI_LP", "LI_HP", "HI_LP", "HI_HP"]], 
    num_rows="fixed",
    use_container_width=True,
    height=350
)

# Update the master school_df
for _, row in type_editor.iterrows():
    sname = row["School_Name"]  
    school_df.loc[school_df["School_Name"] == sname, ["LI_LP", "LI_HP", "HI_LP", "HI_HP"]] = (
        row["LI_LP"], row["LI_HP"], row["HI_LP"], row["HI_HP"]
    )


# =========================================================
# BUILD STUDENTS FROM TYPE COUNTS (must come before r editor)
# =========================================================
np.random.seed(42)
clinician_id = int(clinician_id_input)

# Re-filter schools for this clinician using updated type counts
assigned_schools_raw = school_df.loc[
    school_df["SLP_IDs"].apply(
        lambda ids: clinician_id in ids if isinstance(ids, list) else False
    )
].copy()

if assigned_schools_raw.empty:
    st.error(f"Clinician {clinician_id} has no schools.")
    st.stop()

# reset index
assigned_schools_raw = assigned_schools_raw.reset_index(drop=True)

# ----------------------------------------
# BUILD STUDENTS FROM TYPE COUNTS
# ----------------------------------------
student_school = {}
p = {}  # priority
I = {}  # intensity
students = []  # track valid student IDs
school_type_counts = {}

idx = 0
for j, (_, row) in enumerate(assigned_schools_raw.iterrows()):

    li_lp = int(row["LI_LP"])
    li_hp = int(row["LI_HP"])
    hi_lp = int(row["HI_LP"])
    hi_hp = int(row["HI_HP"])

    # store for pie charts
    school_type_counts[j] = {
        "f1": li_lp,
        "f2": li_hp,
        "f3": hi_lp,
        "f4": hi_hp
    }

    # --- LI-LP (p=2, I=0)
    for _ in range(li_lp):
        student_school[idx] = j
        p[idx] = 2
        I[idx] = 0
        students.append(idx)
        idx += 1

    # --- LI-HP (p=2, I=1)
    for _ in range(li_hp):
        student_school[idx] = j
        p[idx] = 2
        I[idx] = 1
        students.append(idx)
        idx += 1

    # --- HI-LP (p=1, I=0)
    for _ in range(hi_lp):
        student_school[idx] = j
        p[idx] = 1
        I[idx] = 0
        students.append(idx)
        idx += 1

    # --- HI-HP (p=1, I=1)
    for _ in range(hi_hp):
        student_school[idx] = j
        p[idx] = 1
        I[idx] = 1
        students.append(idx)
        idx += 1

students = list(students)

if len(students) == 0:
    st.error("ERROR: No students created. Please fill in LI/HI/HP/LP counts.")
    st.stop()

# ---------------------------------------------------------
#     SIDEBAR — r MATRIX EDITOR (AFTER STUDENTS)
# ---------------------------------------------------------
st.sidebar.header("Enter the referral weeks", divider="gray")

# initialize r values only once
if "r_values" not in st.session_state:
    st.session_state["r_values"] = {}

# ensure r has entries for all students
for i in students:
    if i not in st.session_state["r_values"]:
        st.session_state["r_values"][i] = float(np.random.uniform(-20, -10))

r = st.session_state["r_values"]

# build editable table
r_df = pd.DataFrame({
    "Student": students,
    "r": [r[i] for i in students]
})

edited_r_df = st.sidebar.data_editor(
    r_df,
    num_rows="fixed",
    use_container_width=True,
    height=300
)

# save changes
for _, row in edited_r_df.iterrows():
    r[int(row["Student"])] = float(row["r"])

st.session_state["r_values"] = r

# ---------------------------------------------------------
#     SIDEBAR — SCHOOL SELECTION (AFTER r MATRIX)
# ---------------------------------------------------------
#st.sidebar.header("", divider="gray")

# --- MODIFIED: Use School_Name for selection/display ---
name_to_id = assigned_schools_raw.set_index("School_Name")["School_ID"].to_dict() # Map Name to ID
id_to_name_master = assigned_schools_raw.set_index("School_ID")["School_Name"].to_dict() # Map ID to Name

top8_names = (
    assigned_schools_raw.sort_values("Demand", ascending=False)
    .head(8)["School_Name"].tolist()
)

selected_school_names = st.sidebar.multiselect(
    "Select school names for which you wish to solve the scheduling problem",  
    options=assigned_schools_raw["School_Name"].tolist(),
    default=top8_names
)

# Convert selected names back to IDs for filtering
selected_school_ids = [name_to_id[name] for name in selected_school_names]

# This is the FINAL list of schools the model will use (filtered by ID).
assigned_schools = assigned_schools_raw[
    assigned_schools_raw["School_ID"].isin(selected_school_ids)
].reset_index(drop=True)
# -----------------------------------------------------


# ---------------------------------------------------------
# **FIX: Re-define num_schools and schools list based on the final selection**
# ---------------------------------------------------------
num_schools = len(assigned_schools)
schools = list(range(num_schools))
# ---------------------------------------------------------
# ---------------------------------------------------------


n1_input = st.sidebar.number_input(
    "Enter the number of sessions required for Non-HITS, Low Priority",
    min_value=1, value=8, step=1
)
n2_input = st.sidebar.number_input(
    "Enter the number of sessions required for Non-Hits, High Priority",
    min_value=1, value=10, step=1
)

lp_ratio = st.sidebar.number_input(
    "Enter the ratio for HITS to Non-HITS Sessions (Low Priority)",
    min_value=0.0, value=2.0, step=0.1
)
hp_ratio = st.sidebar.number_input(
    "Enter the ratio for HITS to Non-HITS Sessions (High Priority)",
    min_value=0.0, value=2.0, step=0.1
)

# ---------------------------------------------------------
#     SIDEBAR — CAPACITY + THETA INPUTS
# ---------------------------------------------------------
st.sidebar.header("Capacity & Session Durations", divider="gray")

cap_total = st.sidebar.number_input(
    "Enter TOTAL Weekly Capacity for the Clinician",
    min_value=1,
    value=40,
    step=1
)

theta_low = st.sidebar.number_input(
    "Enter Session Duration for Non-HITS Sessions (default 40)",
    min_value=1,
    value=40,
    step=1
)

theta_high = st.sidebar.number_input(
    "Enter for HITS Sessions (default 7)",
    min_value=1,
    value=7,
    step=1
)

theta_max = st.sidebar.number_input(
    "Enter Availibility of the Clinician (minutes per day)",
    min_value=1,
    value=5 * 60,
    step=5
)

# ---------------------------------------------------------
#     SIDEBAR — PRIORITY RATIO c
# ---------------------------------------------------------
st.sidebar.header("Priority Weight", divider="gray")

priority_ratio_c = st.sidebar.number_input(
    "Enter priority ratio c (weight for HIGH priority vs LOW)",
    min_value=1.0,
    value=5.0,
    step=0.5
)

# ---------------------------------------------------------
#     SIDEBAR — PLANNING HORIZON (target_week)
# ---------------------------------------------------------
st.sidebar.header("Planning Horizon (a default block is 4 months, i.e., 16 weeks)", divider="gray")
planning_horizon = st.sidebar.number_input(
    "Planning horizon (number of weeks in target_week)",
    min_value=1,
    max_value=len(W),
    value=16,
    step=1
)

# ---------------------------------------------------------


# ---------------------------------------------------------
#                   MAIN TITLE
# ---------------------------------------------------------

st.markdown(
    """
    This paragraph describes clinician schedule, caseload overview, demand structure,
    and any additional context related to the optimization framework.
    Replace this text with your final explanation later.
    """
)

# ---------------------------------------------------------
#     COORDINATE LOOKUP (CACHE)
# ---------------------------------------------------------
nomi = pgeocode.Nominatim("CA")
geo = Nominatim(user_agent="clinician_mapping")
coord_cache = {}

def get_coordinates(pc):
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

if "Coordinates" not in school_df.columns or school_df["Coordinates"].isna().any():
    school_df["Coordinates"] = school_df[postal_col].apply(get_coordinates)

# ---------------------------------------------------------
#              CLINICIAN INFORMATION
# ---------------------------------------------------------
st.header("Clinician Information", divider="gray")
st.markdown(
    """
    Now enter the clinician ID for the clinician whose optimal schedule you would like to view. After entering the ID, a map will appear showing the schools assigned to that clinician. On the left, you can also see the demand for each school and adjust it based on the current waitlist in the system. 
    """
)

clinician = slp_df[slp_df["Clinician_ID"] == clinician_id_input]

if clinician.empty:
    st.info("Enter a valid clinician ID.")
    st.stop()

assigned_schools_display = school_df[
    school_df["SLP_IDs"].apply(lambda ids: clinician_id_input in ids if isinstance(ids, list) else False)
]


st.subheader(f"Clinician ID: {clinician_id_input}")

# Get the row for this clinician
clinician_row = slp_df[slp_df["Clinician_ID"] == clinician_id_input]

# Extract schedule (handle case where ID not found)
if not clinician_row.empty:
    schedule = clinician_row["Schedule"].iloc[0]
else:
    schedule = "No schedule found"
    
st.write(
    f"General information: Clinician {clinician_id_input} is assigned to {len(assigned_schools_display)} "
    f"schools and works on **{schedule}**."
)


with st.expander("Show Assigned Schools"):
    for _, row in assigned_schools_display.iterrows():
        # MODIFIED: Show School Name and ID
        st.write(f"• {row['School_Name']} (ID: {row['School_ID']}) — Demand: {row['Demand']}")  

# ---------------------------------------------------------
# PLACEHOLDER PARAGRAPH AFTER CLINICIAN INFORMATION
# ---------------------------------------------------------

# ---------------------------------------------------------
#                INTERACTIVE MAP
# ---------------------------------------------------------
st.header("Clinician School Map", divider="gray")

valid_schools = assigned_schools_display[
    assigned_schools_display["Coordinates"].apply(lambda c: c and c[0] is not None)
]

if valid_schools.empty:
    st.warning("No valid coordinates for this clinician.")
    st.stop()

center_lat = valid_schools["Coordinates"].apply(lambda c: c[0]).mean()
center_lon = valid_schools["Coordinates"].apply(lambda c: c[1]).mean()

fmap = folium.Map(location=[center_lat, center_lon], zoom_start=11)

for _, row in valid_schools.iterrows():
    lat, lon = row["Coordinates"]
    sid = row["School_ID"]
    sname = row["School_Name"] # MODIFIED: Get School Name
    demand_val = row["Demand"]
    postal_val = row[postal_col]

    folium.map.Marker(
        [lat, lon],
        # MODIFIED: Use School Name in marker label
        icon=folium.DivIcon(html=f"<b>{sname}</b>")  
    ).add_to(fmap)

    folium.CircleMarker(
        location=[lat, lon],
        radius=4,
        fill=True,
        color="blue",
        # MODIFIED: Use School Name in popup
        popup=f"School: {sname}<br>ID: {sid}<br>Postal: {postal_val}<br>Demand: {demand_val}"  
    ).add_to(fmap)

st_folium(fmap, width=700, height=450)

# ============================================================
# MODEL PREPARATION (students, types, sessions, r, c, etc.)
# ============================================================


# ----------------------------------------
# SESSIONS
# ----------------------------------------
n1 = int(n1_input)
n2 = int(n2_input)
n3 = int(lp_ratio * n1)
n4 = int(hp_ratio * n2)


# ----------------------------------------
# Filter students based on selected schools
# ----------------------------------------

# Step 1: Create a map from original `School_ID` to new 1-based index (1 to num_schools)
school_id_to_index = {
    row["School_ID"]: idx + 1 for idx, row in assigned_schools.iterrows()
}
# School ID back to the *new* 1-based index
school_idx_to_id = {
    idx + 1: row["School_ID"] for idx, row in assigned_schools.iterrows()
}
# MODIFIED: School Name back to the *new* 1-based index, needed for visualization
school_idx_to_name = {
    idx + 1: row["School_Name"] for idx, row in assigned_schools.iterrows()
}


valid_student_ids = []
new_student_school = {}
j_idx = 0 # New school index 0..num_schools-1
for j_orig, (_, row) in enumerate(assigned_schools_raw.iterrows()): # Iterate over all schools assigned to clinician
    school_id = row["School_ID"]
    
    # Is this school selected by the user?
    if school_id in selected_school_ids:
        # Get the new 1-based index (1 to num_schools)
        j_new = j_idx + 1

        # Check all students that were originally assigned to this school (j_orig index)
        for i in [s_id for s_id in students if student_school[s_id] == j_orig]:
            valid_student_ids.append(i)
            new_student_school[i] = j_new

        j_idx += 1

students = valid_student_ids
student_school = new_student_school # Overwrite with the filtered student set and new school indices

if len(students) == 0:
    st.error("ERROR: No students remain after filtering by selected schools. Please select schools with demand.")
    st.stop()


# Re-calculate parameters for the filtered student set
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
        
    c[i] = priority_ratio_c if p[i] == 1 else 1.0


# ----------------------------------------
# END OF STUDENT/SCHOOL INDEX RE-MAPPING
# ----------------------------------------



# ============================================================
# Define D (range of working days per week) based on schedule
# ============================================================
schedule_days_map = {
    "m-f": 5,
    "tu-th": 3,
    "m-th": 4
}
# Map day index (1=Mon, 2=Tue, ...) to the 1-based index t used in D (1, 2, 3...)
day_index_map = {
    "m-f": [1, 2, 3, 4, 5],
    "tu-th": [2, 3, 4],
    "m-th": [1, 2, 3, 4]
}
weekday_names_map = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri"}


clinician_schedule = (
    slp_df.loc[slp_df["Clinician_ID"] == clinician_id, "Schedule"]
    .squeeze()
)

if pd.isna(clinician_schedule):
    raise ValueError(f"Clinician {clinician_id} has no schedule assigned in slp_df.")

normalized_schedule = clinician_schedule.strip().lower().replace("–", "-")
normalized_schedule = normalized_schedule.replace("m-th", "m-th")

if normalized_schedule not in schedule_days_map:
    raise ValueError(f"Unknown schedule '{clinician_schedule}' for clinician {clinician_id}.")

num_days = schedule_days_map[normalized_schedule]
# D is the 1-based index of working days (e.g., [1, 2, 3] for 3 working days)
D = range(1, num_days + 1)
working_weekdays = day_index_map[normalized_schedule] # [1, 2, 3] for Tue-Thu, where 1=Mon, 2=Tue...

# subset of weeks used in daily objective & caps (user-defined)
target_week = list(range(1, planning_horizon + 1))

# ============================================================
# Travel Time Matrix for Gurobi (not shown on UI)
# ============================================================
coords = assigned_schools["Coordinates"].tolist()
if len(coords) != num_schools:
    raise ValueError("Mismatch in number of schools and coordinates.")

AVERAGE_SPEED = 40  # km/h
KM_TO_MIN = 60 / AVERAGE_SPEED

def euclidean_distance(coord1, coord2):
    if coord1 is None or coord2 is None or None in coord1 or None in coord2:
        return np.nan
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    dlat = (lat2 - lat1) * 111
    dlon = (lon2 - lon1) * 85
    return sqrt(dlat**2 + dlon**2)

# time_matrix indexed by [0..num_schools] for nodes, 0 = depot
time_matrix = np.zeros((num_schools + 1, num_schools + 1))
for i in range(num_schools):
    for j in range(num_schools):
        if i != j:
            dist_km = euclidean_distance(coords[i], coords[j])
            time_matrix[i + 1, j + 1] = round(dist_km * KM_TO_MIN, 1)
for j in range(1, num_schools + 1):
    time_matrix[0, j] = 0
    time_matrix[j, 0] = 0
    
# MODIFIED: Get school names for itinerary (now stores names)
school_names = assigned_schools.set_index(assigned_schools.index + 1)['School_Name'].to_dict()  
school_names[0] = "Office/Depot"
# Get coordinates for the Depot (Centroid of selected schools)
depot_lat = assigned_schools["Coordinates"].apply(lambda c: c[0]).mean()
depot_lon = assigned_schools["Coordinates"].apply(lambda c: c[1]).mean()
depot_coords = (depot_lat, depot_lon)

# FIX: Define nodes and s_idx globally before the optimization/visualization logic needs them
nodes = range(num_schools + 1)  # 0 = depot, 1..|schools|
s_idx = range(1, num_schools + 1)


# ============================================================
# SCHOOL INFORMATION SECTION (PIE CHARTS + r PLOTS)
# ============================================================
st.header("School Information", divider="gray")
st.markdown(
    """
    This section provides details on the schools visited by the clinician.
    We first show the number of students in each category based on the values entered,
    displayed both numerically and in pie charts.

    <br>

    We use the following color scheme for student categories:

    - <span style='color: red; font-weight:600;'>HI-HP</span> = High Intensity, High Priority  
    - <span style='color: orange; font-weight:600;'>HI-LP</span> = High Intensity, Low Priority  
    - <span style='color: blue; font-weight:600;'>LI-HP</span> = Low Intensity, High Priority  
    - <span style='color: green; font-weight:600;'>LI-LP</span> = Low Intensity, Low Priority  

    <br>

    These colors match the pie chart visuals so the clinician can easily interpret the distributions.
    """,
    unsafe_allow_html=True
)

st.subheader("Student Type (based on Priority and Intensity) Distribution per School")
st.markdown(
    """
    This section provides pie charts for the student types. For visual purposes, the default is set on the 8 most populated schools but feel free to change the schools you want to see and optimize over on the left""")

if num_schools == 0:
    st.info("No schools available for school information.")
else:
    cols = st.columns(4)

    for j in schools:
        # Get the School Name from the filtered list (assigned_schools)
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

        col = cols[j % 4]
        with col:
            fig, ax = plt.subplots(figsize=(2.2, 2.2))
            ax.pie(
                sizes,
                labels=["LI-LP", "LI-HP", "HI-LP", "HI-HP"],
                colors=["green", "blue", "orange", "red"], # Match text description
                autopct="%1.0f%%"
            )
            ax.set_title(
                # MODIFIED: Use school name in plot title
                f"{school_name}",  
                fontsize=10
            )
            st.pyplot(fig)


# ---- Paragraph after School Information ----


# ---- r[i] plots by priority and intensity ----
st.subheader("Distribution of the Referral dates")
st.markdown(
    """
    Enter the referral dates for the students on the left. In this section, you can view the distribution of referral dates and compare them across different priority and intensity groups. Note that the start of the planning horizon is defined as week 0, and referral dates are measured relative to this point. Because students are referred before the planning horizon begins, their referral dates appear as negative values. For example, a student referred 3 months before the planning horizon would have a referral date of -3 x 4 = -12 weeks.""")

if len(students) == 0:
    st.info("No students to plot r[i].")
else:
    col1, col2 = st.columns(2)

    # Plot 1: colored by priority (p[i])
    with col1:
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        xs = []
        ys = []
        colors_pr = []
        for i in students:
            xs.append(i)
            ys.append(r[i])
            colors_pr.append("red" if p[i] == 1 else "green")  # red = high priority, green = low
        ax1.scatter(xs, ys, c=colors_pr, s=10)
        ax1.scatter([], [], color="red", label="High Priority")
        ax1.scatter([], [], color="green", label="Low Priority")
        ax1.legend()
        ax1.set_title("Referral date based on the student priority")
        ax1.set_xlabel("Student index")
        ax1.set_ylabel("Referral week")
        st.pyplot(fig1)

    # Plot 2: colored by intensity (I[i])
    with col2:
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        xs2 = []
        ys2 = []
        colors_int = []
        for i in students:
            xs2.append(i)
            ys2.append(r[i])
            colors_int.append("red" if I[i] == 1 else "green")  # red = high intensity, green = low
        ax2.scatter(xs2, ys2, c=colors_int, s=10)
        ax2.scatter([], [], color="red", label="High Intensity")
        ax2.scatter([], [], color="green", label="Low Intensity")
        ax2.legend()
        ax2.set_title("Referral date based on the student intensity")
        ax2.set_xlabel("Student index")
        ax2.set_ylabel("Referral week")
        st.pyplot(fig2)

st.header("Input parameters to the model", divider="gray")
st.markdown(
    """
    Enter the parameters for the model:

    1. **School Selection** Select the schools for which you want to solve the scheduling problem.  
        By default, the app selects the eight most populated schools, but you may change this manually.  
        *Note: selecting more than eight schools may increase solving time.*

    2. **Number of Sessions per Student Type** Enter the number of sessions required for:
        - Normal (non-HITS) students — both low-priority and high-priority  
        - HITS students — based on two learning-curve ratios (low-priority and high-priority)

    3. **Maximum Weekly Sessions** Input the maximum number of total sessions the clinician can handle each week.  
        The optimizer will not exceed this limit.

    4. **Session Durations** Specify the duration of:  
        - **Non-HITS sessions** (default: 40 minutes = 30 session + 10 documentation)  
        - **HITS sessions** (default: 7 minutes = 5 session + 2 documentation)

    5. **Daily Working Time** Enter the clinician’s daily available working minutes (default: 300 minutes = 5 hours).

    6. **Priority Ratio** Provide the weighting factor for high-priority students (both HITS and non-HITS),  
        which determines how strongly the model prioritizes them.

    7. **Planning Horizon** Provide the planning horizon you wish the model to be solved for. (note that usually, not all students can be served in that horizon)
    """
)


st.success("All inputs loaded. Parameters and visualizations computed successfully.")

# ============================================================
# OPTIMIZATION MODEL SECTION (Progress bar + status)
# ============================================================
st.header("Optimization Model", divider="gray")
status_placeholder = st.empty()
progress_bar = st.progress(0)

# ---------------------------------------------------------
#     SIDEBAR — OPTIMIZATION BUTTON (MUST COME FIRST)
# ---------------------------------------------------------

st.sidebar.header("Optimization Control", divider="gray")
# FIX: Define the button variable before the conditional block uses it
start_optimization = st.sidebar.button("Run Optimization")


if start_optimization:
    # Reset success state when optimization starts
    st.session_state.optimization_success = False  
    try:
        st.session_state.optimization_ran = True

        status_placeholder.write("Starting weekly model...")
        progress_bar.progress(5)

        # The model indices are already defined globally above.
        # nodes = range(num_schools + 1)
        # s_idx = range(1, num_schools + 1)

        weekly = gp.Model(f"clinician_{clinician_id}_weekly")

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
                f"Iteration {it}: solving weekly model "
                f"(cap_total={cap_total_value}, max_schools_per_week={max_schools_per_week_value})..."
            )

            weekly.optimize()

            if weekly.status != GRB.OPTIMAL:
                status_placeholder.write("Weekly model infeasible or not optimal. Stopping.")
                break

            # Store weekly results to session state
            st.session_state.x_li_weekly = {(i, w): int(round(x_li_var[i, w].X)) for i in students for w in W}
            st.session_state.aux1_weekly = {(i, w): 3 * int(round(aux1_var[i, w].X)) for i in students for w in W}
            st.session_state.tr_weekly = {(j, w): int(round(tr_var[j, w].X)) for j in s_idx for w in W}

            status_placeholder.write("Weekly model solved. Solving daily subproblem...")
            progress_bar.progress(min(progress + 5, 80))

            m_daily = gp.Model(f"clinician_{clinician_id}_daily_scheduling")
            m_daily.Params.OutputFlag = 0

            x_d   = m_daily.addVars(students, W, D, vtype=GRB.BINARY, name="x")
            v     = m_daily.addVars(s_idx, W, D, vtype=GRB.BINARY, name="v")
            z     = m_daily.addVars(nodes, nodes, W, D, vtype=GRB.BINARY, name="z")
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
                        # An explicit constraint is helpful for LI:
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
                        # And flow must enter j (guaranteed by flow balance above)

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

            status_placeholder.write("Solving daily model...")
            progress_bar.progress(min(progress + 10, 90))
            m_daily.optimize()

            if m_daily.status == GRB.OPTIMAL:
                status_placeholder.write("Daily model feasible. Two-stage solution found.")
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

                # Store the results in session state
                st.session_state.final_daily = m_daily
                st.session_state.final_daily_var = final_daily_var
                st.session_state.optimization_success = True
                break


            if m_daily.status == GRB.INFEASIBLE:
                status_placeholder.write("Daily infeasible → tightening weekly caps...")
                cap_total_value -= CAP_DECREASE
                max_schools_per_week_value -= SCHOOLS_DECREASE

                if cap_total_value < MIN_CAP_TOTAL or max_schools_per_week_value < MIN_SCHOOLS_PER_WEEK:
                    status_placeholder.write("Weekly caps became too tight. Stopping.")
                    break

                for w in W:
                    cap_total_constraints[w].RHS = cap_total_value
                    max_schools_constraints[w].RHS = max_schools_per_week_value

                weekly.update()
                continue

            status_placeholder.write("Daily model status neither OPTIMAL nor INFEASIBLE. Stopping.")
            break

        if st.session_state.optimization_success:
            status_placeholder.write("Optimization complete.")
        else:
            if 'weekly' in locals() and weekly.status == GRB.OPTIMAL: # Check if weekly model was created and optimal
                status_placeholder.write(" No feasible daily solution found for the weekly pattern.")
            else:
                status_placeholder.write("No feasible weekly solution found.")

    except Exception as e:
        status_placeholder.write(f"Error during optimization: {e}")
        progress_bar.progress(0)

#### ---------------------------------------------------------
#     SIDEBAR — VISUALIZATION SETTINGS (AFTER OPTIMIZATION)
# ---------------------------------------------------------

st.sidebar.header("Visualization Settings", divider="gray")

max_weeks_display = st.sidebar.number_input(
    "Maximum number of weeks to display in plots",
    min_value=1,
    max_value=len(W),
    value=40,
    step=1
)

st.sidebar.header("Detailed Visualization Inputs", divider="gray")

# Check if students list is not empty before setting default
default_student_id = students[0] if students else 0

student_id_input = st.sidebar.number_input(
    "Enter Student ID for detailed schedule",
    min_value=0,
    step=1,
    value=default_student_id
)

month_input = st.sidebar.number_input(
    "Enter Month for Calendar View (1–12)",
    min_value=1,
    max_value=12,
    value=10,
    step=1
)

week_input = st.sidebar.number_input(
    "Enter Week for Itinerary/Route Plots",
    min_value=1,
    max_value=199,
    value=1,
    step=1
)

# ---------------------------------------------------------
#     SIDEBAR — OPTIMIZATION BUTTON (LAST)
# ---------------------------------------------------------



# ============================================================
# OPTIMIZATION OUTPUT VISUALIZATION SECTION (BOTTOM)
# ============================================================
st.header("Optimization Output Visualization", divider="gray")

st.markdown(
    """
    This section summarizes the optimized schedule over the planning horizon. On the left, enter the weeks you want the charts to be shown for. Also, you can enter a specific student id to see their weekly and daily schedules. Finally, pick a month for which you want to see the clinician's schedule. 	"""
)

if st.session_state.optimization_ran and st.session_state.optimization_success:
    
    # --- Retrieve data from session state ---
    x_li_weekly = st.session_state.x_li_weekly
    aux1_weekly = st.session_state.aux1_weekly
    final_daily = st.session_state.final_daily
    final_daily_var = st.session_state.final_daily_var
    
    # --- Proceed with visualization ---
    weeks_plot = [w for w in sorted(W) if 1 <= w <= max_weeks_display]

    st.subheader("Number of Appointments by Intensity and Priority")
    st.markdown(
    """
    This section provides the distribution of weekly sessions based on the priority and intensity of the sessions. **Note that each high-intensity week (`HI`) is counted as 3 sessions in the 'High' columns** (matching the total weekly session capacity consumption). 	""")

    weekly_counts = {}
    for w in W:
        weekly_counts[(w, 'Low', 1)] = 0
        weekly_counts[(w, 'Low', 2)] = 0
        weekly_counts[(w, 'High', 1)] = 0
        weekly_counts[(w, 'High', 2)] = 0

    for i in students:
        for w in W:
            if I[i] == 0: # Low Intensity
                if x_li_weekly.get((i, w), 0) > 0:
                    weekly_counts[(w, 'Low', p[i])] += x_li_weekly[(i, w)]
            else: # High Intensity
                if aux1_weekly.get((i, w), 0) > 0:
                    # aux1_weekly stores 3 * binary:
                    weekly_counts[(w, 'High', p[i])] += aux1_weekly[(i, w)]

    # Priority 1: High (HP), Priority 2: Low (LP)
    # LI-LP (p=2, I=0, Green), LI-HP (p=1, I=0, Blue), HI-LP (p=2, I=1, Orange), HI-HP (p=1, I=1, Red)
    
    # Low Priority (p=2) - LP
    low_int_low_pri = [weekly_counts.get((w, 'Low', 2), 0) for w in weeks_plot] # LI-LP
    high_int_low_pri = [weekly_counts.get((w, 'High', 2), 0) for w in weeks_plot] # HI-LP
    
    # High Priority (p=1) - HP
    low_int_high_pri = [weekly_counts.get((w, 'Low', 1), 0) for w in weeks_plot] # LI-HP
    high_int_high_pri = [weekly_counts.get((w, 'High', 1), 0) for w in weeks_plot] # HI-HP


    bar_width = 0.6
    x = np.arange(len(weeks_plot))

    fig, ax = plt.subplots(figsize=(7, 4))
    
    # LI-LP (Green)
    ax.bar(x, low_int_low_pri, color="green", width=bar_width, label="LI-LP")
    
    # LI-HP (Blue) on top of LI-LP
    bottom_li_lp = np.array(low_int_low_pri)
    ax.bar(x, low_int_high_pri, bottom=bottom_li_lp, color="blue", width=bar_width, label="LI-HP")
    
    bottom_li_hp = bottom_li_lp + np.array(low_int_high_pri) # Total LI bottom (Base for HI)

    # HI-LP (Orange) on top of LI total
    bottom_hi_lp = bottom_li_hp # Total LI is the base for HI-LP
    ax.bar(x, high_int_low_pri, bottom=bottom_hi_lp, color="orange", width=bar_width, label="HI-LP")

    # HI-HP (Red) on top of LI total and HI-LP
    bottom_hi_hp = bottom_hi_lp + np.array(high_int_low_pri)
    ax.bar(x, high_int_high_pri, bottom=bottom_hi_hp, color="red", width=bar_width, label="HI-HP")

    ax.set_xlabel("Week", fontsize=11)
    ax.set_ylabel("Number of Weekly Appointments", fontsize=11)
    ax.set_title(f"Number of Appointments by Intensity and Priority", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(weeks_plot, rotation=45)
    ax.legend(loc="upper right", frameon=False)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Number of Appointments Per Week by Intensity and Priority")
    st.markdown(
    """
    This section provides separate effect of priority and intensity on the schedule. 	""")

    col_intensity, col_priority = st.columns(2)

    with col_intensity:
        weekly_low_int = []
        weekly_high_int = []
        for w in weeks_plot:
            low_weighted = sum(x_li_weekly.get((i, w), 0) for i in students if I[i] == 0)
            high_weighted = sum(aux1_weekly.get((i, w), 0) for i in students if I[i] == 1)
            weekly_low_int.append(low_weighted)
            weekly_high_int.append(high_weighted)

        x2 = np.arange(len(weeks_plot))
        width2 = 0.35

        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.bar(x2 - width2 / 2, weekly_low_int, width2, label="Low Intensity (×1)", color="#4fa3ff")
        ax2.bar(x2 + width2 / 2, weekly_high_int, width2, label="High Intensity (×3)", color="#ff6a6a")

        ax2.set_xlabel("Week")
        ax2.set_ylabel("Number of Appointments")
        ax2.set_title("Number of Appointments Per Week by Intensity")
        ax2.set_xticks(x2)
        ax2.set_xticklabels(weeks_plot, rotation=45)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)

    with col_priority:
        weekly_pri1 = []
        weekly_pri2 = []

        for w in weeks_plot:
            pri1_weighted = 0
            pri2_weighted = 0
            for i in students:
                if p[i] == 1: # High Priority
                    pri1_weighted += x_li_weekly.get((i, w), 0) if I[i] == 0 else aux1_weekly.get((i, w), 0)
                else: # Low Priority
                    pri2_weighted += x_li_weekly.get((i, w), 0) if I[i] == 0 else aux1_weekly.get((i, w), 0)
            weekly_pri1.append(pri1_weighted)
            weekly_pri2.append(pri2_weighted)

        x3 = np.arange(len(weeks_plot))
        width3 = 0.35

        fig3, ax3 = plt.subplots(figsize=(6, 3))
        ax3.bar(x3 - width3 / 2, weekly_pri1, width3, label="Priority 1 (High)", color="#34c759")
        ax3.bar(x3 + width3 / 2, weekly_pri2, width3, label="Priority 2 (Low)", color="#ff9500")

        ax3.set_xlabel("Week")
        ax3.set_ylabel("Number of Appointments")
        ax3.set_title("Number of Appointments Per Week by Priority")
        ax3.set_xticks(x3)
        ax3.set_xticklabels(weeks_plot, rotation=45)
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig3)

    # ============================================================
    # EXPANDERS FOR NEW VISUALIZATIONS
    # ============================================================

    # ---------- 1. Student Schedule Plot ----------
    with st.expander("Student Schedule (per student)"):
        def plot_student_schedule(student_id, week_limit=50):
            if student_id not in students:
                st.write("Student ID out of range or student's school not selected for optimization.")
                return
            
            start_week = "N/A (Rerun optimization to calculate)" # Placeholder
            
            if I[student_id] == 0:
                session_weeks = [w for w in W if x_li_weekly.get((student_id, w), 0) > 0]
            else:
                session_weeks = [w for w in W if aux1_weekly.get((student_id, w), 0) > 0]

            intensity = "High" if I[student_id] == 1 else "Low"
            priority = "High" if p[student_id] == 1 else "Low"

            fig_s, ax_s = plt.subplots(figsize=(10, 1.5))
            ax_s.eventplot(session_weeks,
                            colors='tab:blue' if intensity == "Low" else 'tab:red',
                            lineoffsets=0.5,
                            linelengths=0.6)
            ax_s.set_title(f"Student {student_id}: Session Weeks (Intensity={intensity}, Priority={priority}, Start={start_week})")
            ax_s.set_xlabel("Week")
            ax_s.set_yticks([])
            ax_s.set_xlim(1, week_limit)
            ax_s.grid(True, axis='x', linestyle='--', alpha=0.6)
            plt.tight_layout()
            st.pyplot(fig_s)

        st.write("Student schedule for selected student ID.")
        student_id_int = int(student_id_input)
        if student_id_int in students:
            plot_student_schedule(student_id_int, week_limit=max_weeks_display)
        else:
            st.warning(f"Student ID {student_id_int} is not in the set of students generated for the selected schools.")


    # ---------- 2. Daily Session Calendar Plot ----------
    st.write("Daily schedule (which days within each week the student has sessions).")

    def plot_student_daily_calendar(student_id):
        if student_id not in students:
            st.write("Student ID out of range or student's school not selected for optimization.")
            return

        student_days = [
            (w, d)
            for (i, w, d), val in final_daily_var.items()
            if i == student_id and val > 0.5
        ]

        if not student_days:
            st.info("No daily sessions found for this student.")
            return

        fig, ax = plt.subplots(figsize=(12, 4))

        # Map the model's 1-based index (t in D) back to the actual weekday index (1=Mon, 2=Tue...)
        model_day_to_weekday_index = {t: working_weekdays[t-1] for t in D}
        
        # Plot points: xs=Week, ys=Weekday Index
        xs = [w for (w, d) in student_days]
        ys = [model_day_to_weekday_index[d] for (w, d) in student_days]

        intensity_color = "red" if I[student_id] == 1 else "blue"

        ax.scatter(xs, ys, color=intensity_color, s=70)

        # Set axis formatting
        weekday_labels = [weekday_names_map[i] for i in working_weekdays]
        weekday_positions = working_weekdays
        
        ax.set_yticks(weekday_positions)  
        ax.set_yticklabels(weekday_labels)
        ax.set_xlabel("Week")
        ax.set_ylabel("Day of Week")
        ax.set_title(f"Daily Appointment Calendar — Student {student_id}")

        ax.grid(True, linestyle="--", alpha=0.5)

        st.pyplot(fig)

    if student_id_int in students:
        plot_student_daily_calendar(student_id_int)
    else:
        pass


    # ---------- 3. Monthly Calendar View ----------
    with st.expander("Monthly Calendar View (Visits & HI/LI)"):
        def build_date_mapping(W, D, year=2026):
            
            # Find the Monday of Week 1 of the planning horizon
            first = datetime.date(year, 1, 1)
            # Find the first Monday of the year (2026/01/05)
            first_monday = first + datetime.timedelta(days=(7 - first.weekday()) % 7)  
            
            mapping = {}
            W_sorted = sorted(W)
            
            # Map the model's t (1, 2, 3...) to the actual day of the week (1=Mon, 2=Tue...)
            model_day_to_weekday_index = {t: working_weekdays[t-1] for t in D}
            
            for w_idx, w in enumerate(W_sorted):
                for t in D:
                    # t is the model's day index (1, 2, 3...)
                    weekday_index = model_day_to_weekday_index[t]  
                    
                    # Calculate day offset: 
                    # w_idx * 7 = start of week w (as Monday)
                    # (weekday_index - 1) = offset from Monday to the actual working day
                    day_offset = w_idx * 7 + (weekday_index - 1)
                    
                    mapped_date = first_monday + datetime.timedelta(days=day_offset)
                    mapping[(w, t)] = mapped_date
                    
            return mapping

        date_map = build_date_mapping(W, D, year=2026)

        def plot_month_calendar(year, month):
            day_info = {}
            
            # Map new 1-based index to actual School Name (MODIFIED)
            local_school_idx_to_name = {  
                idx + 1: assigned_schools.loc[idx, "School_Name"]  
                for idx in assigned_schools.index
            }

            for (w, t), date in date_map.items():
                if date.year != year or date.month != month:
                    continue

                schools_today = set()
                HI_count = 0
                LI_count = 0
                
                for i in students:
                    # Need to retrieve the variable object by name from the stored model, then get its value
                    var_x = final_daily.getVarByName(f"x[{i},{w},{t}]")
                    if var_x is not None and var_x.X > 0.5:
                        if I[i] == 1:
                            HI_count += 1
                        else:
                            LI_count += 1
                        
                        # Use student_school map to find the school index j (1-based)
                        j = student_school[i]
                        # MODIFIED: Get School Name
                        schools_today.add(str(local_school_idx_to_name.get(j, f"S_Err_{j}")))


                if HI_count + LI_count > 0: # Only record days with actual sessions planned
                    day_info[date.day] = {
                        "schools": sorted(list(schools_today)),
                        "HI": HI_count,  
                        "LI": LI_count
                    }

            cal = calendar.Calendar(firstweekday=0)  # Monday
            month_grid = cal.monthdayscalendar(year, month)

            fig_c, ax_c = plt.subplots(figsize=(14, 9))
            ax_c.set_axis_off()

            day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            for col, name in enumerate(day_names):
                ax_c.text(col + 0.5, 1, name, fontsize=15, ha="center", va="bottom")

            for row, week in enumerate(month_grid):
                for col, day in enumerate(week):
                    x0 = col
                    y0 = -row - 1

                    if day != 0:
                        date_obj = datetime.date(year, month, day)
                        weekday = date_obj.weekday() # 0=Mon, 6=Sun
                    else:
                        weekday = None

                    # Determine day type for background color
                    rect_color = "white"
                    is_scheduled_day = weekday is not None and (weekday + 1) in working_weekdays
                    
                    if weekday in [5, 6]: # Sat/Sun
                             rect_color = "#f0f0f0" # Light grey for weekends
                    elif day != 0 and not is_scheduled_day:
                         # Non-scheduled working days (e.g., Mon/Fri if schedule is Tu-Thu)
                             rect_color = "#f0f8ff" # Light blue
                    
                    if day in day_info: # Day with scheduled sessions
                             rect_color = "#c8f7c5" # Light green

                    ax_c.add_patch(plt.Rectangle((x0, y0), 1, 1, facecolor=rect_color, edgecolor="black", zorder=0))

                    if day != 0:
                        ax_c.text(x0 + 0.05, y0 + 0.8, str(day), fontsize=12, weight="bold")

                        if day in day_info:
                            info = day_info[day]
                            schools = info["schools"]
                            HI_count = info["HI"]
                            LI_count = info["LI"]

                            school_str = ", ".join(schools) if schools else "N/A"
                            # MODIFIED: Display School Names
                            txt = f"Schools: {school_str}\nHI:{HI_count}  LI:{LI_count}"  
                            ax_c.text(x0 + 0.05, y0 + 0.55, txt, fontsize=10, va="top")
                        elif is_scheduled_day:
                             # Keep blank for scheduled working days without activity
                             pass


            ax_c.set_title(f"{calendar.month_name[month]} {year} – Clinician Schedule", fontsize=22)
            ax_c.set_xlim(0, 7)
            ax_c.set_ylim(-len(month_grid) - 1, 1.5)
            plt.tight_layout()
            st.pyplot(fig_c)

        st.write(" In this section you can visualize the monthly schedule of the clinician. Each day features the number of schools visited and the number of students visited based on the session intensity .")
        plot_month_calendar(2026, int(month_input))

    # ---------- 4. Travels Per Day (given week) - RESTORED PLOT ----------
    with st.expander("Travels Per Day in a Given Week"):
        def plot_travels_per_day(week):
            day_labels = []
            travel_counts = []
            for t in sorted(D):
                daily_travel = 0.0
                for j in nodes:
                    for k in nodes:
                        if j == k:
                            continue
                        # Use the final_daily model for z variable values
                        var = final_daily.getVarByName(f"z[{j},{k},{week},{t}]")
                        if var is not None and var.X > 0.5: # Use > 0.5 to count binary flows
                            daily_travel += var.X
                
                day_name = weekday_names_map.get(working_weekdays[t-1])
                day_labels.append(f"{day_name} (Day {t})")
                travel_counts.append(daily_travel)

            fig_t, ax_t = plt.subplots(figsize=(8, 5))
            ax_t.bar(day_labels, travel_counts)
            ax_t.set_title(f"Number of Travels per Day (including from and to office)– Week {week}")
            ax_t.set_xlabel("Day")
            ax_t.set_ylabel("Number of Travels")
            ax_t.grid(axis="y", alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig_t)

        st.write("The number of travels within each week.")
        plot_travels_per_day(int(week_input))

    # ---------- 5. Therapy vs Travel Time Ratio ----------
    with st.expander("Therapy vs Travel Time Ratio per Week"):
        therapy_time = {w: 0.0 for w in W}
        travel_time = {w: 0.0 for w in W}

        for w in W:
            for t in D:
                for i in students:
                    var_x = final_daily.getVarByName(f"x[{i},{w},{t}]")
                    if var_x is not None and var_x.X > 0.5:
                        therapy_time[w] += theta[i]
                for j in nodes:
                    for k in nodes:
                        if j != k:
                            var_z = final_daily.getVarByName(f"z[{j},{k},{w},{t}]")
                            if var_z is not None and var_z.X > 0.5:
                                travel_time[w] += time_matrix[j, k]

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
            ax_r.plot(valid_weeks, valid_ratios, marker="o")
            ax_r.set_xlim(valid_weeks[0], valid_weeks[-1])  
        
        ax_r.set_xlabel("Week")
        ax_r.set_ylabel("Therapy Time / (Therapy + Travel Time)")
        ax_r.set_title("Weekly Ratio of Therapy Time to Total Time")
        ax_r.set_ylim(0, 1)
        ax_r.grid(True)
        plt.tight_layout()
        st.pyplot(fig_r)

    # ---------- 6. Daily Route Network (Itinerary + Map) ----------
    with st.expander("Daily Itinerary and Route Map"):
        
        @st.cache_data(show_spinner=False, max_entries=50) # Cache the heavy path reconstruction
        # FIX: Added leading underscore to _final_daily_model to bypass caching hash check
        def reconstruct_path_for_day(week, day, _final_daily_model, nodes, time_matrix_data, num_schools):
            arcs = set()
            for j in nodes:
                for k in nodes:
                    if j == k:
                        continue
                    var = _final_daily_model.getVarByName(f"z[{j},{k},{week},{day}]")
                    if var is not None and var.X > 0.5:
                        arcs.add((j, k))
            
            if not arcs:
                return None, None
            
            path = []
            current = 0
            
            while current != 0 or not path:  
                
                # Find the next node (k) and the travel time
                nxt_info = next(((k, time_matrix_data[current, k]) for (j, k) in arcs if j == current), None)
                
                if nxt_info is None:
                    if current == 0:  
                        path = []  
                    break
                
                nxt, travel_time = nxt_info
                
                path.append((current, travel_time))

                arcs.discard((current, nxt))
                current = nxt
            
            if current == 0 and path and path[-1][0] != 0:
                path.append((0, 0))
                
            nodes_path = [p[0] for p in path]
            travel_times = [p[1] for p in path[:-1]]
            
            return nodes_path, travel_times  

        # Helper to get sessions (not cached because it's fast)
        def get_sessions_at_school(j, week, day):
            sessions = []
            school_name = school_idx_to_name.get(j)  
            if not school_name:
                return sessions

            students_at_school = [i for i, school_j in student_school.items() if school_j == j]
            
            for i in students_at_school:
                var_x = final_daily.getVarByName(f"x[{i},{week},{day}]") # CORRECTED: Changed {w} and {t} to {week} and {day}
                if var_x is not None and var_x.X > 0.5:
                    intensity = "HI" if I[i] == 1 else "LI"
                    priority = "HP" if p[i] == 1 else "LP"
                    sessions.append({
                        "id": i,
                        "type": f"{intensity}-{priority}",
                        "duration": theta[i] # This is the therapy time in minutes
                    })
            return sessions


        def print_day_itinerary(week, day):
            
            nodes_path, travel_times = reconstruct_path_for_day(
                week, day, final_daily, nodes, time_matrix, num_schools
            )
            
            day_name = weekday_names_map.get(working_weekdays[day-1])
            
            st.markdown(f"### 🗓️ Week {week}, {day_name} (Day {day}) Itinerary")

            if not nodes_path:
                st.info("No visits scheduled for this day.")
                return

            path = nodes_path
            itinerary = []
            
            # --- Text Itinerary ---
            
            # Use school_names (which now contains names)
            itinerary.append(f"**Start**: Leave {school_names.get(path[0], 'Depot')}")  
            
            processed_schools = set()
            
            for i in range(len(path) - 1):
                j = path[i] # Current node
                k = path[i+1] # Next node
                
                # Use school_names (which now contains names)
                j_name = school_names.get(j, f"School Index {j}")  
                k_name = school_names.get(k, f"School Index {k}")  
                travel_time_min = travel_times[i]
                
                # Travel step
                if j != k:
                    if k != 0:
                        itinerary.append(f"**Travel**: {j_name} → {k_name} ({travel_time_min:.1f} minutes)")
                    else:
                        itinerary.append(f"**End**: Return to {k_name} ({travel_time_min:.1f} minutes travel)")

                # Activity at next node k (if it's a school and hasn't been processed this iteration)
                if k != 0 and k not in processed_schools:
                    sessions = get_sessions_at_school(k, week, day)
                    if sessions:
                        total_duration = sum(s['duration'] for s in sessions)
                        # Display School Name and ID
                        itinerary.append(f"**Arrive at {k_name} (ID: {school_idx_to_id.get(k)})**: {len(sessions)} session(s) scheduled (Total therapy time: **{total_duration:.0f} minutes**).")  
                        
                        for s in sessions:
                            # THIS LINE IS THE FIX: IT INCLUDES THE DURATION
                            itinerary.append(f"   - Session: Student **{s['id']}** ({s['type']}) (**{s['duration']:.0f} min**)")
                        
                        processed_schools.add(k)
                    else:
                        # Diagnostic message 
                        itinerary.append(f"**Arrive at {k_name} (ID: {school_idx_to_id.get(k)})**: No sessions scheduled on this specific day/week combination.")
                        processed_schools.add(k)
                        
            st.markdown("\n".join(itinerary))
            
            st.markdown("---")
            
            # --- Map Visualization ---
            st.markdown("### 🗺️ Daily Route Map")
            
            # Get coords for all visited schools and Depot
            route_coords = {}
            for node_idx in set(path):
                if node_idx == 0:
                    route_coords[0] = depot_coords
                else:
                    # Find the school row in the filtered assigned_schools by index (node_idx - 1)
                    school_df_index = node_idx - 1
                    if 0 <= school_df_index < len(assigned_schools):
                        school_row = assigned_schools.iloc[school_df_index]
                        route_coords[node_idx] = school_row["Coordinates"]
            
            # Center map on the centroid
            daily_fmap = folium.Map(location=[depot_lat, depot_lon], zoom_start=11)
            
            # Draw Route Lines
            valid_points = [route_coords[node] for node in path if node in route_coords and route_coords[node] is not None]
            if valid_points:
                folium.PolyLine(valid_points, color="red", weight=2.5, opacity=1).add_to(daily_fmap)
            
            # Mark Depot
            folium.Marker(
                depot_coords,
                icon=folium.Icon(color='green', icon='briefcase', prefix='fa'),
                popup="**Depot (Centroid)**"
            ).add_to(daily_fmap)

            # Mark Schools in Order
            # Get only unique school stops (excluding depot)
            unique_stops = [node for node in path if node != 0]
            
            for idx, node_idx in enumerate(unique_stops, start=1):
                if node_idx in route_coords and route_coords[node_idx] is not None:
                    lat, lon = route_coords[node_idx]
                    sid = school_idx_to_id.get(node_idx)
                    sname = school_idx_to_name.get(node_idx)  
                    
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=8,
                        color='blue',
                        fill=True,
                        fill_color='blue',
                        fill_opacity=0.7,
                        # Update popup to show Name
                        popup=f"School **{sname}** (ID: {sid}) (Stop #{idx})"  
                    ).add_to(daily_fmap)
                    
                    # MODIFIED BLOCK: Show School Name instead of stop index
                    folium.Marker(
                        [lat, lon],
                        icon=folium.DivIcon(
                            html=f'<div style="font-size: 10pt; color: black; font-weight: bold; white-space: nowrap; transform: translate(-50%, -150%);">{sname}</div>'
                        )
                    ).add_to(daily_fmap)


            st_folium(daily_fmap, width=700, height=450, key=f"map_{week}_{day}")
            

        st.write("Textual itinerary and visual route map for each working day in the selected week.")
        
        for t in sorted(D):
            print_day_itinerary(int(week_input), t)

else:
    st.info("Run the optimization to see the output visualizations.")