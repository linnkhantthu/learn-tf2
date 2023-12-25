import tensorflow as tf
from tensorflow.python.trackable.autotrackable import AutoTrackable


def predict(model_path: str, Object: object) -> any:
    """_summary_

    Args:
        model_path (str): Path to the model 
        Object (object): Object instance of the model (input)

    Returns:
        probability: Probability likely to be 1
    """
    # Load model
    imported = tf.saved_model.load(model_path)

    # Create an example of a standard proto storing data for training and inference.
    example = tf.train.Example()
    # Storing features into example proto
    for feature_name in Object.__dict__:
        example.features.feature[feature_name].bytes_list.value.extend(
            [Object.__dict__[feature_name].encode()])

    # Predicting
    result = imported.signatures["predict"](
        examples=tf.constant([example.SerializeToString()]))

    # Finally reaturning a probability of having diabetes
    return result["probabilities"]._numpy()[0][1]
