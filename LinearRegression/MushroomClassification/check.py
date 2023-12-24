import tensorflow as tf
from sys import argv
import os

# Load model path
path = os.path.dirname(os.path.abspath(__file__))
imported = tf.saved_model.load(
    f"{path}/model/1703403827/")


def predict(cap_shape, cap_surface, cap_color, bruises,
            odor, gill_attachment, gill_spacing, gill_size, gill_color,
            stalk_shape, stalk_root, stalk_surface_above_ring,
            stalk_surface_below_ring, stalk_color_above_ring, stalk_color_below_ring,
            veil_type, veil_color, ring_number, ring_type, spore_print_color, population, habitat, *args):
    """_summary_

    Args:
        cap_shape (_type_): _description_
        cap_surface (_type_): _description_
        cap_color (_type_): _description_
        bruises (_type_): _description_
        odor (_type_): _description_
        gill_attachment (_type_): _description_
        gill_spacing (_type_): _description_
        gill_size (_type_): _description_
        gill_color (_type_): _description_
        stalk_shape (_type_): _description_
        stalk_root (_type_): _description_
        stalk_surface_above_ring (_type_): _description_
        stalk_surface_below_ring (_type_): _description_
        stalk_color_above_ring (_type_): _description_
        stalk_color_below_ring (_type_): _description_
        veil_type (_type_): _description_
        veil_color (_type_): _description_
        ring_number (_type_): _description_
        ring_type (_type_): _description_
        spore_print_color (_type_): _description_
        population (_type_): _description_
        habitat (_type_): _description_

    Returns:
        _type_: _description_
    """

    # Column Names
    COLUMN_NAMES = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
                    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
                    'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
                    'stalk-surface-below-ring', 'stalk-color-above-ring',
                    'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
                    'ring-type', 'spore-print-color', 'population', 'habitat']

    # Create an example of a standard proto storing data for training and inference.
    example = tf.train.Example()

    # Creating a dataframe dict
    dfpredict = {
        'cap-shape': cap_shape, 'cap-surface': cap_surface, 'cap-color': cap_color, 'bruises': bruises, 'odor': odor,
        'gill-attachment': gill_attachment, 'gill-spacing': gill_spacing, 'gill-size': gill_size, 'gill-color': gill_color,
        'stalk-shape': stalk_shape, 'stalk-root': stalk_root, 'stalk-surface-above-ring': stalk_surface_above_ring,
        'stalk-surface-below-ring': stalk_surface_below_ring, 'stalk-color-above-ring': stalk_color_above_ring,
        'stalk-color-below-ring': stalk_color_below_ring, 'veil-type': veil_type, 'veil-color': veil_color, 'ring-number': ring_number,
        'ring-type': ring_type, 'spore-print-color': spore_print_color, 'population': population, 'habitat': habitat
    }

    print(example.features.feature.keys())

    # Storing features into example proto
    for feature_name in COLUMN_NAMES:
        example.features.feature[feature_name].bytes_list.value.extend(
            [dfpredict[feature_name].encode()])

    # Predicting
    result = imported.signatures["predict"](
        examples=tf.constant([example.SerializeToString()]))

    # Finally reaturning a probability of having diabetes
    return result["probabilities"]._numpy()[0][1]


print(predict(*argv[1:]))
