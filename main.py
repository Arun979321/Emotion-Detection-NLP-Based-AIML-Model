import streamlit as st
import subprocess

# Ensure joblib is installed
subprocess.run(["pip", "install", "joblib"])

import joblib  # ✅ Import after ensuring it's installed

# Optional: show installed packages (for debugging)
installed = subprocess.getoutput("pip list")
st.text(installed)

# Load trained model and tools
model = joblib.load('emotion_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Prediction function
def predict_emotion(text):
    vec = vectorizer.transform([text])
    label = model.predict(vec)[0]
    return label_encoder.inverse_transform([label])[0]

# UI
st.title("Emotion Classifier")
st.write("Enter your message and let the model guess the emotion!")

user_input = st.text_area("Enter a message:")

if st.button("Predict Emotion"):
    if user_input.strip():
        emotion = predict_emotion(user_input)
        st.success(f"Predicted Emotion: **{emotion}**")
    else:
        st.warning("Please enter some text.")
