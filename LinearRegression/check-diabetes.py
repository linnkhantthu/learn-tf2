import tensorflow as tf
from sys import argv
import os

# Load model path
path = os.path.dirname(os.path.abspath(__file__))
imported = tf.saved_model.load(
    f"{path}/model/model_diabetes_females/")


def predict(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, *args):
    """Predict Function
    Example command: python check-diabetes.py 1 85 66 29 0 26.6 0.351 31

    Keyword arguments:
    Pregnancies: Number of times pregnant
    Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
    BloodPressure: Diastolic blood pressure (mm Hg)
    SkinThickness: Triceps skin fold thickness (mm)
    Insulin: 2-Hour serum insulin (mu U/ml)
    BMI: Body mass index (weight in kg/(height in m)^2)
    DiabetesPedigreeFunction: Diabetes pedigree function
    Age: Age (years)
    Return: Probability of the patient likely to have diabetes
    """

    # Turning args string to float
    Pregnancies = float(Pregnancies)
    Glucose = float(Glucose)
    BloodPressure = float(BloodPressure)
    SkinThickness = float(SkinThickness)
    Insulin = float(Insulin)
    BMI = float(BMI)
    DiabetesPedigreeFunction = float(DiabetesPedigreeFunction)
    Age = float(Age)

    # Column Names
    COLUMN_NAMES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

    # Create an example of a standard proto storing data for training and inference.
    example = tf.train.Example()

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
