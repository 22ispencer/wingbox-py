from wingbox_py.geometry import Geometry


class Section(Geometry):
    elastic_mod: float

    def __init__(self, elastic_mod: float) -> None:
        super().__init__()
        self.elastic_mod = elastic_mod

def mod_weighted_centroid(sections: list[Section]):
