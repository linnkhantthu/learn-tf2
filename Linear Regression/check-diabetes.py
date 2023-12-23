import tensorflow as tf
from sys import argv
import os

path = os.path.dirname(os.path.abspath(__file__))
imported = tf.saved_model.load(
    f"{path}/model/model_diabetes_females/")


def predict(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, *args):
    COLUMNS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
               "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    example = tf.train.Example()

    Pregnancies = float(Pregnancies)
    Glucose = float(Glucose)
    BloodPressure = float(BloodPressure)
    SkinThickness = float(SkinThickness)
    Insulin = float(Insulin)
    BMI = float(BMI)
    DiabetesPedigreeFunction = float(DiabetesPedigreeFunction)
    Age = float(Age)

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
    for feature_name in COLUMNS:
        example.features.feature[feature_name].float_list.value.extend(
            dfpredict[feature_name])

    print(example)

    result = imported.signatures["predict"](
        examples=tf.constant([example.SerializeToString()]))
    # def input_fn(data_df, batch_size=32):
    #     return tf.data.Dataset.from_tensor_slices(dict(data_df)).batch(batch_size)

    # r = imported.signatures["serving_default"](
    #     tf.constant([dfpredict], dtype=tf.float32))
    # print(r)

    return result["probabilities"]._numpy()[0][1]


print(predict(*argv[1:]))

# result = predict(1, 85, 66, 29, 0, 26.6, 0.351, 31)
# print(result)
