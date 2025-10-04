import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline
model = joblib.load("best_xgb_pipeline.pkl")

# -----------------------------
# Numeric mapping for EmpEducationLevel
# -----------------------------
education_mapping = {
    'Below College': 0,
    'College': 1,
    'Bachelor': 2,
    'Master': 3,
    'Doctor': 4
}

# -----------------------------
# Known categories for categorical columns (from training)
# -----------------------------
known_categories = {
    'EducationBackground': ['College', 'Bachelor', 'Master'],  # categories seen during training
    'Gender': ['Male', 'Female'],
    'MaritalStatus': ['Single', 'Married', 'Divorced'],
    'EmpDepartment': ['Sales', 'Research & Development', 'HR'],
    'EmpJobRole': [
        'Sales Executive', 'Research Scientist', 'Laboratory Technician',
        'Manufacturing Director', 'Healthcare Representative',
        'Manager', 'Sales Representative', 'Research Director', 'Human Resources'
    ],
    'BusinessTravelFrequency': ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'],
    'OverTime': ['Yes', 'No'],
    'Attrition': ['Yes', 'No']
}

# -----------------------------
# Helper function for safe categorical mapping
# -----------------------------
def safe_category(value, feature_name):
    """Return the value if known, else fallback to first known category"""
    return value if value in known_categories[feature_name] else known_categories[feature_name][0]

# -----------------------------
# Streamlit inputs
# -----------------------------
st.title("Employee Performance Prediction")

# Numeric inputs
age = st.number_input("Age", 18, 70, 30)
distance = st.number_input("Distance from Home (km)", 0, 100, 10)
emp_env_satisfaction = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
emp_hourly_rate = st.number_input("Hourly Rate", 0, 1000, 50)
emp_job_involvement = st.slider("Job Involvement (1-4)", 1, 4, 3)
emp_job_level = st.slider("Job Level", 1, 5, 2)
emp_job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
num_companies_worked = st.number_input("Number of Companies Worked", 0, 20, 1)
last_salary_hike_percent = st.number_input("Last Salary Hike %", 0, 100, 10)
emp_relationship_satisfaction = st.slider("Relationship Satisfaction (1-4)", 1, 4, 3)
total_work_experience = st.number_input("Total Work Experience (Years)", 0, 50, 5)
training_last_year = st.number_input("Training Times Last Year", 0, 20, 2)
work_life_balance = st.slider("Work Life Balance (1-4)", 1, 4, 3)
years_at_company = st.number_input("Years at This Company", 0, 50, 3)
years_in_current_role = st.number_input("Years in Current Role", 0, 50, 2)
years_since_last_promotion = st.number_input("Years Since Last Promotion", 0, 50, 1)
years_with_manager = st.number_input("Years with Current Manager", 0, 50, 2)

# Categorical inputs
gender = st.selectbox("Gender", known_categories['Gender'])
education_level = st.selectbox("Education Level", list(education_mapping.keys()))
marital_status = st.selectbox("Marital Status", known_categories['MaritalStatus'])
emp_department = st.selectbox("Department", known_categories['EmpDepartment'])
emp_job_role = st.selectbox("Job Role", known_categories['EmpJobRole'])
business_travel = st.selectbox("Business Travel Frequency", known_categories['BusinessTravelFrequency'])
overtime = st.selectbox("OverTime", known_categories['OverTime'])
attrition = st.selectbox("Attrition", known_categories['Attrition'])

# -----------------------------
# Prepare input DataFrame with safe mapping
# -----------------------------
input_dict = {
    'Age': [age],
    'DistanceFromHome': [distance],
    'EmpEducationLevel': [education_mapping[education_level]],
    'EmpEnvironmentSatisfaction': [emp_env_satisfaction],
    'EmpHourlyRate': [emp_hourly_rate],
    'EmpJobInvolvement': [emp_job_involvement],
    'EmpJobLevel': [emp_job_level],
    'EmpJobSatisfaction': [emp_job_satisfaction],
    'NumCompaniesWorked': [num_companies_worked],
    'EmpLastSalaryHikePercent': [last_salary_hike_percent],
    'EmpRelationshipSatisfaction': [emp_relationship_satisfaction],
    'TotalWorkExperienceInYears': [total_work_experience],
    'TrainingTimesLastYear': [training_last_year],
    'EmpWorkLifeBalance': [work_life_balance],
    'ExperienceYearsAtThisCompany': [years_at_company],
    'ExperienceYearsInCurrentRole': [years_in_current_role],
    'YearsSinceLastPromotion': [years_since_last_promotion],
    'YearsWithCurrManager': [years_with_manager],
    # Categorical safely mapped
    'Gender': [safe_category(gender, 'Gender')],
    'EducationBackground': [safe_category(education_level, 'EducationBackground')],
    'MaritalStatus': [safe_category(marital_status, 'MaritalStatus')],
    'EmpDepartment': [safe_category(emp_department, 'EmpDepartment')],
    'EmpJobRole': [safe_category(emp_job_role, 'EmpJobRole')],
    'BusinessTravelFrequency': [safe_category(business_travel, 'BusinessTravelFrequency')],
    'OverTime': [safe_category(overtime, 'OverTime')],
    'Attrition': [safe_category(attrition, 'Attrition')]
}

X_input = pd.DataFrame(input_dict)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Performance"):
    try:
        prediction = model.predict(X_input)
        st.success(f"Predicted Performance: {prediction[0]}")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
