import streamlit as st
import pandas as pd
import numpy as np
import joblib

pipeline = joblib.load("notebooks/model_pipeline.pkl")
model = joblib.load("notebooks/final_model.pkl")
target_encoder = joblib.load("notebooks/target_encoder.pkl")

st.set_page_config(page_title = "Income Classifier", layout= "centered")
st.title("💰 Income Classifier App")
st.markdown("---")
st.write("Predict whether income is <=50K or >50K")

st.sidebar.header("Enter User Details")


education_map = {
    "Preschool": 1,
    "1st-4th": 2,
    "5th-6th": 3,
    "7th-8th": 4,
    "9th": 5,
    "10th": 6,
    "11th": 7,
    "12th": 8,
    "HS-grad": 9,
    "Some-college": 10,
    "Assoc-voc": 11,
    "Assoc-acdm": 12,
    "Bachelors": 13,
    "Masters": 14,
    "Prof-school": 15,
    "Doctorate": 16
    }

age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=38)

workclass = st.sidebar.selectbox("WorkClass", options=['State-gov','Self-emp-not-inc','Private','Federal-gov','Local-gov','Unknown','Self-emp-inc','Without-pay','Never-worked'])

education = st.sidebar.selectbox("Education", options=["Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th", "HS-grad", "Some-college", "Assoc-voc", "Assoc-acdm", "Bachelors", "Masters", "Prof-school", "Doctorate"])

education_num = education_map[education]

marital_status = st.sidebar.selectbox("Marital_Status", options=['Never-married','Married-civ-spouse','Divorced','Married-spouse-absent','Separated','Married-AF-spouse','Widowed'])

occupation	= st.sidebar.selectbox("Occupation", options=['Adm-clerical','Exec-managerial','Handlers-cleaners','Prof-specialty','Other-service','Sales','Craft-repair','Transport-moving','Farming-fishing','Machine-op-inspct','Tech-support','Unknown','Protective-serv','Armed-Forces','Priv-house-serv'])
        
relationship = st.sidebar.selectbox("Relationship", options=['Not-in-family','Husband','Wife','Own-child','Unmarried','Other-relative'])

gender = st.sidebar.selectbox("Gender", options=['Male','Female'])
        
capital_gain = st.sidebar.number_input("Capital_gain", min_value=0, value=0)
        
capital_loss = st.sidebar.number_input("Capital_loss", min_value=0, value=0)
        
hours_per_week	= st.sidebar.number_input("Hours_per_week", min_value=1, max_value=100, value=40)

native_country = st.sidebar.selectbox("Native_country", options=['United-States','Cuba','Jamaica','India','Unknown','Mexico','South','Puerto-Rico','Honduras','England','Canada','Germany','Iran','Philippines','Italy','Poland','Columbia','Cambodia','Thailand','Ecuador','Laos','Taiwan','Haiti', 
                                                                 'Portugal','Dominican-Republic','El-Salvador','France','Guatemala','China','Japan','Yugoslavia','Peru','Outlying-US(Guam-USVI-etc)','Scotland','Trinadad&Tobago','Greece','Nicaragua','Vietnam','Hong','Ireland','Hungary','Holand-Netherlands'])
                                                                 
if st.sidebar.button("Predict Income"):

    input_data = pd.DataFrame({
        "age": [age],
        "workclass": [workclass],
        "education": [education],
        "education_num": [education_num],
        "marital_status": [marital_status],	
        "occupation": [occupation],
        "relationship": [relationship],
        "gender": [gender], 
        "capital_gain": [capital_gain],
        "capital_loss": [capital_loss],
        "hours_per_week": [hours_per_week],
        "native_country": [native_country]
    })

    input_data['capital_gain'] = np.log1p(input_data['capital_gain'])
    input_data['capital_loss'] = np.log1p(input_data['capital_loss'])

    transformed_data = pipeline.transform(input_data)

    prediction = model.predict(transformed_data )

    probability = model.predict_proba(transformed_data)[0][1]

    final_prediction = target_encoder.inverse_transform(prediction)[0]

    st.subheader("Prediction Result")

    if final_prediction == ">50K":
        st.success(f"Predicted Income Class: {final_prediction}")
    else:
        st.warning(f"Predicted Income Class: {final_prediction}")

    st.metric(
        label="Probability of Income >50K",
        value=f"{probability:.2%}"
    )
    
st.markdown("---")
st.caption("Built using Streamlit and CatBoost")
    











