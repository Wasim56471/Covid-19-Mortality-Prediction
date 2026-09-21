"""
COVID-19 Mortality Prediction: ML Project Demo

Gradio interface for a logistic regression model trained to predict
COVID-19 mortality from age, gender, vaccination status, symptoms and
pre-existing conditions.

This app is a machine learning project demonstration and is not
intended for medical purposes.

Run locally:
    pip install -r requirements.txt
    python app.py
"""

import gradio as gr
import joblib
import pandas as pd

# The bundle holds the full Pipeline (scaler + classifier), the exact
# feature order used in training, and the chosen decision threshold.
bundle = joblib.load("covid_model.joblib")

SYMPTOM_LIST = ["Fever", "Cough", "Fatigue", "Shortness of breath",
                "Loss of smell", "Headache"]
CONDITION_LIST = ["Diabetes", "Hypertension", "Heart disease",
                  "Asthma", "Cancer"]

DISCLAIMER = (
    "### ⚠️ Not for medical use\n\n"
    "This app is a machine learning project demonstration. It is not a "
    "medical tool and must not be used to make any decision about health, "
    "diagnosis or treatment. If you are unwell, contact a doctor or NHS 111."
)


def predict(age, gender, vaccination, symptoms, conditions):
    # Rebuild the feature row in exactly the training order.
    # Reference categories are Female and Booster Dose, represented
    # by all of their dummy columns being 0.
    row = pd.DataFrame([{
        "age": age,
        "symptom_count": len(symptoms),
        "comorbidity_count": len(conditions),
        "gender_Male": 1.0 if gender == "Male" else 0.0,
        "gender_Other": 1.0 if gender == "Other" else 0.0,
        "vaccination_status_Fully Vaccinated":
            1.0 if vaccination == "Fully Vaccinated" else 0.0,
        "vaccination_status_Partially Vaccinated":
            1.0 if vaccination == "Partially Vaccinated" else 0.0,
        "vaccination_status_Unvaccinated":
            1.0 if vaccination == "Unvaccinated" else 0.0,
    }])[bundle["feature_order"]]

    risk = bundle["model"].predict_proba(row)[0, 1]
    flagged = risk >= bundle["threshold"]

    headline = (
        f"## Predicted mortality risk: **{risk:.1%}**\n\n"
        f"Based on patterns in the training data."
    )

    if flagged:
        verdict = (
            "**The model puts this patient in its higher-risk group.**\n\n"
            "That group is about 1 in 5 of everyone. The model catches roughly "
            "5 out of every 6 deaths by looking at this group, but most people "
            "in it survive: only about 1 in 6 flagged patients actually died."
        )
    else:
        verdict = (
            "**The model puts this patient in its lower-risk group.**\n\n"
            "That group is about 4 in 5 of everyone. Most deaths are not in "
            "this group, but the model still misses roughly 1 death in 6, so "
            "being here is not a guarantee of anything."
        )

    return headline + "\n\n" + verdict + "\n\n---\n\n" + DISCLAIMER


with gr.Blocks(title="COVID-19 Mortality Prediction: ML Demo") as demo:
    gr.Markdown(
        "# COVID-19 Mortality Prediction: ML Project Demo\n\n"
        "**This app is a machine learning project demonstration and is not "
        "intended for medical purposes.** Enter patient details to see how a "
        "trained logistic regression model produces a prediction."
    )

    with gr.Row():
        with gr.Column():
            age = gr.Slider(0, 99, value=45, step=1, label="Age")
            gen = gr.Radio(["Female", "Male", "Other"],
                           value="Female", label="Gender")
            vac = gr.Radio(["Booster Dose", "Fully Vaccinated",
                            "Partially Vaccinated", "Unvaccinated"],
                           value="Fully Vaccinated", label="Vaccination status")
            sym = gr.CheckboxGroup(SYMPTOM_LIST, label="Symptoms present")
            con = gr.CheckboxGroup(CONDITION_LIST,
                                   label="Pre-existing conditions")
            btn = gr.Button("Run model", variant="primary")
        with gr.Column():
            out = gr.Markdown()

    btn.click(predict, [age, gen, vac, sym, con], out)

    with gr.Accordion("Technical details", open=False):
        gr.Markdown(
            "### Model\n"
            "Logistic regression on 8 features: age, symptom count, comorbidity "
            "count, gender (2 dummy columns) and vaccination status (3 dummy "
            "columns). Features standardised inside a scikit-learn Pipeline. "
            "Trained on 3,000 patients with 131 deaths (4.37%).\n\n"
            "### Performance (held-out test set, 750 patients, 33 deaths)\n"
            "| Metric | Value |\n|---|---|\n"
            "| ROC-AUC | 0.866 (SD 0.015 over 30 splits) |\n"
            "| PR-AUC | 0.167 (SD 0.021), no-skill baseline 0.044 |\n"
            "| Brier score | 0.0403, base-rate benchmark 0.0421 |\n"
            "| Decision threshold | 0.06 |\n"
            "| Sensitivity | 0.848 (28 of 33 deaths caught) |\n"
            "| Specificity | 0.801 |\n"
            "| Precision | 0.164 (171 flagged, 28 died) |\n\n"
            "### Known limitations of this model\n"
            "- Age has very little effect on its predictions "
            "(odds ratio 1.008 per year).\n"
            "- It assigns children a higher risk than young adults.\n"
            "- The output has no defined time horizon, as the training data "
            "has no follow-up period.\n"
            "- Predictions above about 20% are unverified, since very few "
            "patients in the data scored that high."
        )

if __name__ == "__main__":
    demo.launch()
