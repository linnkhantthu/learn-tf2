import tensorflow as tf
from sys import argv
import os

# Load model path
path = os.path.dirname(os.path.abspath(__file__))
imported = tf.saved_model.load(
    f"{path}/model/model_diabetes_females/")


def predict(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, *args):
    """Predict Function

    Keyword arguments:
    Pregnancies -- Number of pregnancies that the patient carried
    Glucose -- Glucose level of the patient
    BloodPressure -- BloodPressure of the patient
    SkinThickness -- Skin thickness of the arm from the back
    Insulin -- Insulin of the patient
    BMI -- BMI of the patient
    DiabetesPedigreeFunction -- DiabetesPedigreeFunction
    Age -- Age
    Return: Probability of the patient to have diabetes
    """

    # Column Names
    COLUMN_NAMES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    # Create an example of a standard proto storing data for training and inference.
    example = tf.train.Example()

    # Turning args string to float
    Pregnancies = float(Pregnancies)
    Glucose = float(Glucose)
    BloodPressure = float(BloodPressure)
    SkinThickness = float(SkinThickness)
    Insulin = float(Insulin)
    BMI = float(BMI)
    DiabetesPedigreeFunction = float(DiabetesPedigreeFunction)
    Age = float(Age)

    # Creating a dataframe dict
    dfpredict = {
        "Pregnancies": [Pregnancies],
        "Glucose": [Glucose],
        "BloodPressure": [BloodPressure],
        "SkinThickness": [SkinThickness],
        "Insulin": [Insulin],
        "BMI": [BMI],
        "DiabetesPedigreeFunction": [DiabetesPedigreeFunction],
        "Age": [Age],
    }

    # Storing features into example proto
    for feature_name in COLUMN_NAMES:
        example.features.feature[feature_name].float_list.value.extend(
            dfpredict[feature_name])

    # Predicting
    result = imported.signatures["predict"](
        examples=tf.constant([example.SerializeToString()]))

    # Finally reaturning a probability of having diabetes
    return result["probabilities"]._numpy()[0][1]


print(predict(*argv[1:]))

# eg command: python check-diabetes.py 1 85 66 29 0 26.6 0.351 31
