from viktor import ViktorController


class Controller(ViktorController):
    """Controller for project folder"""

    label = "Projects"
    children = ["Project"]
    show_children_as = "Table"
