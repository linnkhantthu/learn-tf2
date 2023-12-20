import tensorflow as tf
from sys import argv

imported = tf.saved_model.load(
    "/home/linn/Projects/tf2/model_diabetes/")


def predict(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, *args):

    Pregnancies = float(Pregnancies)
    Glucose = float(Glucose)
    BloodPressure = float(BloodPressure)
    SkinThickness = float(SkinThickness)
    Insulin = float(Insulin)
    BMI = float(BMI)
    DiabetesPedigreeFunction = float(DiabetesPedigreeFunction)
    Age = float(Age)

    example = tf.train.Example()
    example.features.feature["Pregnancies"].float_list.value.extend([
                                                                    Pregnancies])
    example.features.feature["Glucose"].float_list.value.extend([Glucose])
    example.features.feature["BloodPressure"].float_list.value.extend([
                                                                      BloodPressure])
    example.features.feature["SkinThickness"].float_list.value.extend([
                                                                      SkinThickness])
    example.features.feature["Insulin"].float_list.value.extend([Insulin])
    example.features.feature["BMI"].float_list.value.extend([BMI])
    example.features.feature["DiabetesPedigreeFunction"].float_list.value.extend([
                                                                                 DiabetesPedigreeFunction])
    example.features.feature["Age"].float_list.value.extend([Age])
    result = imported.signatures["predict"](
        examples=tf.constant([example.SerializeToString()]))

    return result["probabilities"]._numpy()[0][1]


print(predict(*argv[1:]))
# result = predict(1, 85, 66, 29, 0, 26.6, 0.351, 31)

# print(result)
