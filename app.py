import os
import gradio as gr
import joblib

# Get the absolute path to the model
MODEL_PATH = "model/drug_pipeline.joblib"

# Check if the model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

# Load trained model
pipe = joblib.load(MODEL_PATH)

# Prediction function
def predict_drug(age, gender, blood_pressure, cholesterol, na_to_k_ratio):
    features = [age, gender, blood_pressure, cholesterol, na_to_k_ratio]
    predicted_drug = pipe.predict([features])[0]
    return f"Predicted Drug: {predicted_drug}"

# Input widgets
inputs = [
    gr.Slider(15, 74, step=1, label="Age"),
    gr.Radio(["M", "F"], label="Gender"),
    gr.Radio(["HIGH", "LOW", "NORMAL"], label="Blood Pressure"),
    gr.Radio(["HIGH", "NORMAL"], label="Cholesterol"),
    gr.Slider(6.2, 38.2, step=0.1, label="Na_to_K"),
]

# Output widget
outputs = gr.Label()

# Example inputs
examples = [
    [30, "M", "HIGH", "NORMAL", 15.4],
    [35, "F", "LOW", "NORMAL", 8],
    [50, "M", "HIGH", "HIGH", 34],
]

# Launch Gradio interface
gr.Interface(
    fn=predict_drug,
    inputs=inputs,
    outputs=outputs,
    examples=examples,
    title="Drug Classification",
    description="Enter the details to identify the drug type",
    theme=gr.themes.Soft(),
).launch()