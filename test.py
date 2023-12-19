import tensorflow as tf
import os
imported = tf.saved_model.load(
    "/home/linn/Projects/tf2/1702964658/")


def predict(Pregnancies,	Glucose,	BloodPressure,	SkinThickness,	Insulin,	BMI,	DiabetesPedigreeFunction,	Age):
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
    return result["probabilities"]


print(predict(1, 85, 66, 29, 0, 26.6, 0.351, 31,))
