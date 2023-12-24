import os
import tensorflow as tf
from sys import argv

# Load model path
path = os.path.dirname(os.path.abspath(__file__))
imported = tf.saved_model.load(
    f"{path}/model/1703389724/")


def predict(SepalLength, SepalWidth, PetalLength, PetalWidth, *args):

    # Column Names
    COLUMN_NAMES = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]
    SepalLength = float(SepalLength)
    SepalWidth = float(SepalWidth)
    PetalLength = float(PetalLength)
    PetalWidth = float(PetalWidth)

    example = tf.train.Example()

    dfpredict = {
        "SepalLength": [SepalLength],
        "SepalWidth": [SepalWidth],
        "PetalLength": [PetalLength],
        "PetalWidth": [PetalWidth]
    }

    for feature_name in COLUMN_NAMES:
        example.features.feature[feature_name].float_list.value.extend(
            dfpredict[feature_name]
        )
    result = imported.signatures["predict"](
        examples=tf.constant([example.SerializeToString()])
    )
    class_id = result["class_ids"]._numpy()[0][0]
    print(class_id)
    return result["probabilities"]._numpy()[0][class_id]


print(predict(*argv[1:]))
