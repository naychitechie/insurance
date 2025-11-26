import streamlit as st
from pycaret.regression import load_model, predict_model
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Insurance App", layout="wide")

def load_model_wrapper():
    load_error = None
    try:
        model = load_model('dt_insurance_charges_model')
    except Exception as e:
        model = None
        load_error = e
        st.sidebar.error(
            "Model failed to load. If you see an OSError mentioning 'libomp', install libomp (Homebrew: `brew install libomp`) or use conda: `conda install -c conda-forge libomp`. Full error printed in the main area."  
        )
        st.error(f"Model load error: {e}")
    return model

def predict(model, input_df):
    predictions_df = predict_model(model, data=input_df)
    predictions = predictions_df.iloc[0]['prediction_label']
    return predictions


def run():
    st.title("Insurance Charges Prediction App")
    st.sidebar.header("Input Features")
    image= Image.open('logo.png')
    image_hospital = Image.open('hospital.jpeg')

    st.image(image)
    st.sidebar.info("This app is created to predict patient hospital charges")
    st.sidebar.image(image_hospital)
    # Try to load the trained model here (lazy load) so the app can start
    # even if a native dependency like libomp is missing; show a friendly
    # message in the sidebar if loading fails.
    add_selectbox = st.sidebar.selectbox("How would you like to predict?", ("Online", "Batch"))

    model = load_model_wrapper()

    if add_selectbox == "Online":
        st.sidebar.subheader("Patient Data")
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        sex = st.selectbox("Sex", ['Male', 'Female'])
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)

        if st.checkbox("Smoker"):
            smoker = 'Yes'
        else:
            smoker = 'No'
        
        region = st.selectbox("Region", ['Northeast', 'Northwest', 'Southeast', 'Southwest'])
        output = ""

        input_dict = {'age': age, 'sex': sex, 'bmi': bmi,
                      'children': children, 'smoker': smoker,
                      'region': region}
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            if model is None:
                st.error("Model is not loaded. See the sidebar for details and install instructions.")
            else:
                output = predict(model=model, input_df=input_df)
                st.success(f'The predicted insurance charge is: {output}')
    else:
        st.subheader("Batch Prediction")
        file_upload = st.file_uploader("Upload CSV file for predictions", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            if model is None:
                st.error("Model is not loaded. See the sidebar for details and install instructions.")
            else:
                predictions = predict_model(estimator=model, data=data)
                st.success("Predictions:")
                st.write(predictions)
    # age = st.sidebar.slider("Age", 18, 100, 30)
    # bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
    # children = st.sidebar.slider("Number of Children

if __name__ == '__main__':
    run()