import streamlit as st
import pickle
#import numpy as np

# Load the saved model
with open("titanic_predictor.sav", "rb") as file:
    model = pickle.load(file)

# Function to make predictions
def predict_survival(model, features):
    prediction = model.predict([features])
    return "Survived" if prediction[0] == 1 else "Did Not Survive"

# Streamlit app
def main():
    st.title("Titanic Survival Predictor")
    st.write("Enter the details below to predict survival on the Titanic:")

    # Input fields
    Gender = st.radio("Gender", ("Male", "Female"))
    Age = st.number_input("Age", min_value=0, max_value=100, step=1, value=30)
    Pclass = st.selectbox("Ticket Class (Pclass)", [1, 2, 3])
    SibSp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, step=1, value=0)
    Parch = st.number_input("Number of Parents/Children Aboard", min_value=0, step=1, value=0)
    Fare = st.number_input("Fare (in USD)", min_value=0.0, step=0.01, value=50.0)

    # Convert gender to numeric
    Sex = 1 if Gender == "Female" else 0

    # Prepare features for prediction
    features = [Pclass, Sex, Age, SibSp, Parch, Fare]
    #features_array = np.array(features)
    #single_sample = features_array.reshape(1,-1)
    
    # Prediction
    if st.button("Predict"):
        result = predict_survival(model, features)
        st.write(f"Prediction: {result}")

if __name__ == "__main__":
    main()
