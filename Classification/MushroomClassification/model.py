class Mushroom:
    cap_shape: str
    cap_surface: str
    cap_color: str
    bruises: str
    odor: str
    gill_attachment: str
    gill_spacing: str
    gill_size: str
    gill_color: str
    stalk_shape: str
    stalk_root: str
    stalk_surface_above_ring: str
    stalk_surface_below_ring: str
    stalk_color_above_ring: str
    stalk_color_below_ring: str
    veil_type: str
    veil_color: str
    ring_number: str
    ring_type: str
    spore_print_color: str
    population: str
    habitat: str

    def __init__(self, cap_shape: str,
                 cap_surface: str,
                 cap_color: str,
                 bruises: str,
                 odor: str,
                 gill_attachment: str,
                 gill_spacing: str,
                 gill_size: str,
                 gill_color: str,
                 stalk_shape: str,
                 stalk_root: str,
                 stalk_surface_above_ring: str,
                 stalk_surface_below_ring: str,
                 stalk_color_above_ring: str,
                 stalk_color_below_ring: str,
                 veil_type: str,
                 veil_color: str,
                 ring_number: str,
                 ring_type: str,
                 spore_print_color: str,
                 population: str,
                 habitat: str):
        self.cap_shape = cap_shape
        self.cap_surface = cap_surface
        self.cap_color = cap_color
        self.bruises = bruises
        self.odor = odor
        self.gill_attachment = gill_attachment
        self.gill_spacing = gill_spacing
        self.gill_size = gill_size
        self.gill_color = gill_color
        self.stalk_shape = stalk_shape
        self.stalk_root = stalk_root
        self.stalk_surface_above_ring = stalk_surface_above_ring
        self.stalk_surface_below_ring = stalk_surface_below_ring
        self.stalk_color_above_ring = stalk_color_above_ring
        self.stalk_color_below_ring = stalk_color_below_ring
        self.veil_type = veil_type
        self.veil_color = veil_color
        self.ring_number = ring_number
        self.ring_type = ring_type
        self.spore_print_color = spore_print_color
        self.population = population
        self.habitat = habitat
