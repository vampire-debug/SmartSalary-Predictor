import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

import os
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
file_path = os.path.join(BASE_DIR, "dataset", "salary_prediction.csv")

data = pd.read_csv(file_path)

# Encode text columns
label_encoders = {}

categorical_columns = [
    "job_title",
    "education_level",
    "industry",
    "company_size",
    "location",
    "remote_work"
]

for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Features and target
X = data[[
    "job_title",
    "experience_years",
    "education_level",
    "skills_count",
    "industry",
    "company_size",
    "location",
    "remote_work",
    "certifications"
]]

y = data["salary"]

# Train model
model = RandomForestRegressor(
    n_estimators=20,
    max_depth=10,
    random_state=42
)
model.fit(X, y)

# Streamlit UI
st.title("Employee Salary Prediction")

job_title = st.selectbox(
    "Job Title",
    label_encoders["job_title"].classes_
)

experience = st.slider(
    "Years of Experience",
    0,
    30,
    1
)

education = st.selectbox(
    "Education Level",
    label_encoders["education_level"].classes_
)

skills = st.slider(
    "Skills Count",
    1,
    20,
    5
)

industry = st.selectbox(
    "Industry",
    label_encoders["industry"].classes_
)

company_size = st.selectbox(
    "Company Size",
    label_encoders["company_size"].classes_
)

location = st.selectbox(
    "Location",
    label_encoders["location"].classes_
)

remote = st.selectbox(
    "Remote Work",
    label_encoders["remote_work"].classes_
)

certifications = st.slider(
    "Certifications",
    0,
    10,
    1
)

# Prediction button
if st.button("Predict Salary"):

    input_data = [[
        label_encoders["job_title"].transform([job_title])[0],
        experience,
        label_encoders["education_level"].transform([education])[0],
        skills,
        label_encoders["industry"].transform([industry])[0],
        label_encoders["company_size"].transform([company_size])[0],
        label_encoders["location"].transform([location])[0],
        label_encoders["remote_work"].transform([remote])[0],
        certifications
    ]]

    prediction = model.predict(input_data)

    st.success(f"Predicted Salary: ₹{prediction[0]:,.2f}")