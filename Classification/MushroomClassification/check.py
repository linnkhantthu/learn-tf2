import os
from Classification.MushroomClassification.model import Mushroom
from utils import predict
from sys import argv


def main(cap_shape, cap_surface, cap_color, bruises,
         odor, gill_attachment, gill_spacing, gill_size, gill_color,
         stalk_shape, stalk_root, stalk_surface_above_ring,
         stalk_surface_below_ring, stalk_color_above_ring, stalk_color_below_ring,
         veil_type, veil_color, ring_number, ring_type, spore_print_color, population, habitat, *args):

    # Model Path
    model_path = os.path.dirname(
        os.path.abspath(__file__)) + "/model/1703662515/"

    # Creating Model Object
    mushroom = Mushroom(cap_shape, cap_surface, cap_color, bruises,
                        odor, gill_attachment, gill_spacing, gill_size, gill_color,
                        stalk_shape, stalk_root, stalk_surface_above_ring,
                        stalk_surface_below_ring, stalk_color_above_ring, stalk_color_below_ring,
                        veil_type, veil_color, ring_number, ring_type, spore_print_color, population, habitat)

    return predict(model_path, mushroom)


print(main(*argv[1:]))
