list_of_rules = ["color_equal", "position_equal", "size_equal"]


def color_equal(obj1_colors, obj2_colors):
    """
    checks if the two objects passed are the same color
    the reason why we pass in a list of colors for each object instead of just 1 color:
    in future implementation there will be objects that are multi-colored
    :param obj1_colors: list of colors from object 1
    :param obj2_colors: list of colors from object 2
    """
    return obj1_colors == obj2_colors


def position_equal(obj1_pixels, obj2_pixels):
    """
    checks if two objects have the same position
    :param obj1_pixels: list of pixel coordinates from object 1 (unsorted)
    :param obj2_pixels: list of pixel coordinates from object 2 (unsorted)
    :return:
    """
    return obj1_pixels == obj2_pixels


def size_equal(obj1_size, obj2_size):
    """
    checks if the two objects have the same size
    :param obj1_size: size of obj1
    :param obj2_size: size of obj2
    """
    return obj1_size == obj2_size

