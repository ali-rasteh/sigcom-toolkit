import math


def get_visible_points(width, length, ref_x, ref_y, direction_deg, alpha_deg, d):
    """
    Calculates points visible from an arbitrary reference point in the room.
    Origin (0,0) is at the upper-right corner.

    Parameters:
    - width (float): The width of the room (horizontal axis, x goes from -width to 0).
    - length (float): The length of the room (vertical axis, y goes from -length to 0).
    - ref_x (float): The X coordinate of the reference point.
    - ref_y (float): The Y coordinate of the reference point.
    - direction_deg (float): The central direction of sight in degrees.
    - alpha_deg (float): The field of view spreads +/- this many degrees from the direction.
    - d (float): The distance granularity (step size).

    Returns:
    - List of tuples (x, y) representing visible points, sorted closest to furthest.
    """

    # Check if the reference point itself is inside the room
    if not (-width <= ref_x <= 0 and -length <= ref_y <= 0):
        raise ValueError("The reference point is outside the bounds of the room.")

    # Convert angles to radians
    direction = math.radians(direction_deg)
    alpha = math.radians(alpha_deg)

    # Maximum possible distance is the furthest corner from the reference point
    corners = [
        (0, 0),                     # Upper-right
        (-width, 0),                # Upper-left
        (0, -length),               # Lower-right
        (-width, -length)           # Lower-left
    ]
    max_distance = max(math.hypot(ref_x - cx, ref_y - cy) for cx, cy in corners)

    visible_points = []

    # Start iterating from distance 'd' outwards
    r = d
    while r <= max_distance:
        # Calculate angular step to keep spatial distance roughly equal to 'd'
        angular_step = d / r

        # Sweep from the left edge of the FOV to the right edge
        theta = direction - alpha

        while theta <= direction + alpha + 1e-9:
            # Convert polar to Cartesian, offset by the reference point's location
            x = ref_x + r * math.cos(theta)
            y = ref_y + r * math.sin(theta)

            # Check if the generated point is actually inside the room
            if -width <= x <= 0 and -length <= y <= 0:
                visible_points.append((round(x, 4), round(y, 4)))

            theta += angular_step

        r += d

    return visible_points

def get_viewing_angle_range(width, length, ref_x, ref_y, obj_x, obj_y, alpha_deg):
    """
    Calculates the range of central direction angles at a reference point
    required to see an object point in the room.

    Parameters:
    - width (float): The width of the room.
    - length (float): The length of the room.
    - ref_x, ref_y (float): Coordinates of the reference point.
    - obj_x, obj_y (float): Coordinates of the target object point.
    - alpha_deg (float): The field of view spread (+/- degrees).

    Returns:
    - A tuple: (min_angle, max_angle, exact_angle) in degrees.
    """

    # Check if both points are inside the room
    if not (-width <= ref_x <= 0 and -length <= ref_y <= 0):
        raise ValueError("The reference point is outside the bounds of the room.")
    if not (-width <= obj_x <= 0 and -length <= obj_y <= 0):
        raise ValueError("The object point is outside the bounds of the room.")

    # Calculate differences in coordinates
    dx = obj_x - ref_x
    dy = obj_y - ref_y

    # If the reference point and object are at the exact same location
    if dx == 0 and dy == 0:
        return 0.0, 360.0, 0.0 # It can be seen looking in any direction

    # math.atan2(y, x) returns the angle in radians between -pi and pi
    exact_angle_rad = math.atan2(dy, dx)
    exact_angle_deg = math.degrees(exact_angle_rad)

    # Normalize the exact angle to be strictly between 0 and 360 degrees
    exact_angle_deg = exact_angle_deg % 360

    # The valid central viewing directions are exactly +/- alpha from the exact angle
    min_angle = exact_angle_deg - alpha_deg
    max_angle = exact_angle_deg + alpha_deg

    return round(min_angle, 2), round(max_angle, 2), round(exact_angle_deg, 2)

