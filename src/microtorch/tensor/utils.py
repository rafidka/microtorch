def identify_broadcasting_dimensions(
    shape1: tuple[int, ...], shape2: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Identifies the dimensions along which broadcasting will occur for each shape.

    Args:
        shape1 (tuple[int, ...]): The shape of the first tensor.
        shape2 (tuple[int, ...]): The shape of the second tensor.

    Returns:
        tuple[tuple[int, ...], tuple[int, ...]]: A tuple of two shapes, the first shape
            is where broadcasting will occur for the first tensor, and the second shape
            is where broadcasting will occur for the second tensor.

    Raises:
        ValueError: If the shapes are incompatible for broadcasting.
    """
    # Align shapes from the right by padding shorter shape with 1s
    len_diff = len(shape1) - len(shape2)
    if len_diff > 0:
        # shape2 is shorter, pad with 1s on the left
        padded_shape2 = (-1,) * len_diff + shape2
        padded_shape1 = shape1
    elif len_diff < 0:
        # shape1 is shorter, pad with 1s on the left
        padded_shape1 = (-1,) * (-len_diff) + shape1
        padded_shape2 = shape2
    else:
        # Same length, no padding needed
        padded_shape1 = shape1
        padded_shape2 = shape2

    shape1_broadcasting: list[int] = []
    shape2_broadcasting: list[int] = []

    for i, (dim1, dim2) in enumerate(zip(padded_shape1, padded_shape2)):
        if dim1 == -1:
            # -1 is the special value for padding above, so definitely broadcasting
            shape1_broadcasting.append(i)
        elif dim2 == -1:
            # -1 is the special value for padding above, so definitely broadcasting
            shape2_broadcasting.append(i)
        elif dim1 == 1 and dim2 != 1:
            # Dimension 1 is 1, so broadcasting
            shape1_broadcasting.append(i)
        elif dim1 != 1 and dim2 == 1:
            # Dimension 2 is 1, so broadcasting
            shape2_broadcasting.append(i)
        elif dim1 != 1 and dim2 != 1 and dim1 != dim2:
            # Both dimensions are not 1, so broadcasting is not possible
            raise ValueError(
                f"Incompatible shapes for broadcasting: {shape1} and {shape2}. "
                f"Dimension {i} has sizes {dim1} and {dim2}, which are different and neither is 1."
            )

    return tuple(shape1_broadcasting), tuple(shape2_broadcasting)
