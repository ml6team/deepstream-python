def rect_params_to_coords(rect_params):
    top = int(rect_params.top)
    left = int(rect_params.left)
    width = int(rect_params.width)
    height = int(rect_params.height)

    x1 = left
    y1 = top
    x2 = left + width
    y2 = top + height

    return x1, y1, x2, y2
