import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="pal27/tourism_model", filename="best_tourism_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism Package App")
st.write("""
This application predicts the likelihood of a  customer purchasing the Wellness tourism package.
Please enter the data below to get a prediction.
""")

# User input
TypeofContact = st.selectbox("TypeofContact", ["Self Enquiry", "Company Invited"])
Occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
Gender = st.selectbox("Gender", ["Female", "Male", "Fe Male"])
ProductPitched = st.selectbox("ProductPitched", [ "Deluxe", ",Basic", ",Standard", ",Super Deluxe", ",King"])
MaritalStatus = st.selectbox("MaritalStatus", ["Single", ",Divorced", ",Married", ",Unmarried"])
Designation = st.selectbox("Designation", ["Manager", ",Executive", ",Senior Manager", ",AVP", ",VP"])
Age = st.number_input("Age", min_value=18.0, max_value=100.0)
CityTier = st.number_input("CityTier", min_value=1, max_value=3)
DurationOfPitch = st.number_input("DurationOfPitch", min_value=5, max_value=200)
NumberOfPersonVisiting = st.number_input("NumberOfPersonVisiting", min_value=1, max_value=10)
NumberOfFollowups = st.number_input("NumberOfFollowups", min_value=1, max_value=10)
PreferredPropertyStar = st.number_input("PreferredPropertyStar", min_value=1.0, max_value=5.0)
NumberOfTrips = st.number_input("NumberOfTrips", min_value=1.0, max_value=50.0)
Passport = st.number_input("Passport", min_value=0, max_value=1)
PitchSatisfactionScore = st.number_input("PitchSatisfactionScore", min_value=1, max_value=10)
OwnCar = st.number_input("OwnCar", min_value=0, max_value=10)
NumberOfChildrenVisiting = st.number_input("NumberOfChildrenVisiting", min_value=0, max_value=10)
MonthlyIncome = st.number_input("MonthlyIncome", min_value=1000, max_value=100000)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'TypeofContact': TypeofContact,
    'Occupation': Occupation,
    'Gender': Gender,
    'ProductPitched': ProductPitched,
    'MaritalStatus': MaritalStatus,
    'Designation': Designation,
    'Age': Age,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome
}])


if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Product Purchased" if prediction == 1 else "Not Purchased"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
