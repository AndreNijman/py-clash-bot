import time

import cv2
import numpy as np

from pyclashbot.emulators.capture import FrameData

from .template_cache import TEMPLATE_CACHE

# =============================================================================
# IMAGE RECOGNITION FUNCTIONS
# =============================================================================


def find_image(
    frame: FrameData,
    folder: str,
    tolerance: float = 0.88,
    subcrop: tuple[int, int, int, int] | None = None,
    show_image: bool = False,
) -> tuple[int, int] | None:
    """Find the first matching reference image in a frame."""

    if not isinstance(frame, FrameData):
        if isinstance(frame, np.ndarray):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = FrameData(
                bgr=frame,
                scaled_bgr=frame,
                gray=gray,
                timestamp=time.perf_counter(),
                downscale=1.0,
                original_shape=frame.shape,
                scaled_shape=frame.shape,
            )
        else:
            raise TypeError("find_image expects a FrameData instance or numpy array")

    coord = TEMPLATE_CACHE.find(frame, folder, tolerance, subcrop)
    if coord is not None and show_image:
        print(f"Template match found for {folder} at {coord}")
    return coord


def find_references(
    frame: FrameData,
    folder: str,
    tolerance: float = 0.88,
) -> tuple[list[list[int] | None], list[str]]:
    """Compatibility wrapper that delegates to the cached template lookup."""

    if not isinstance(frame, FrameData):
        if isinstance(frame, np.ndarray):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = FrameData(
                bgr=frame,
                scaled_bgr=frame,
                gray=gray,
                timestamp=time.perf_counter(),
                downscale=1.0,
                original_shape=frame.shape,
                scaled_shape=frame.shape,
            )
        else:
            raise TypeError("find_references expects a FrameData instance or numpy array")
    coord = TEMPLATE_CACHE.find(frame, folder, tolerance, None)
    if coord is None:
        return [None], [folder]
    return [[coord[1], coord[0]]], [folder]


def compare_images(
    image: np.ndarray,
    template: np.ndarray,
    threshold=0.8,
):
    """Detects pixel location of a template in an image using template matching

    Args:
        image (numpy.ndarray): image to find template within
        template (numpy.ndarray): template image to match to
        threshold (float, optional): matching threshold. Defaults to 0.8

    Returns:
        list[int] | None: pixel location [y, x] or None if not found
    """
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)

    # Check if template is larger than image
    if template_gray.shape[0] > img_gray.shape[0] or template_gray.shape[1] > img_gray.shape[1]:
        return None

    res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)

    return None if len(loc[0]) != 1 else [int(loc[0][0]), int(loc[1][0])]


# =============================================================================
# PIXEL RECOGNITION FUNCTIONS
# =============================================================================


def pixel_is_equal(
    pix1: tuple[int, int, int] | list[int],
    pix2: tuple[int, int, int] | list[int],
    tol: float,
) -> bool:
    """Check if two pixels are equal within tolerance

    Args:
    ----
        pix1: first RGB pixel
        pix2: second RGB pixel
        tol: color tolerance

    Returns:
    -------
        bool: whether pixels are equal within tolerance

    """
    diff_r = abs(int(pix1[0]) - int(pix2[0]))
    diff_g = abs(int(pix1[1]) - int(pix2[1]))
    diff_b = abs(int(pix1[2]) - int(pix2[2]))
    return (diff_r < tol) and (diff_g < tol) and (diff_b < tol)


def check_line_for_color(
    emulator,
    x_1: int,
    y_1: int,
    x_2: int,
    y_2: int,
    color: tuple[int, int, int],
) -> bool:
    """Check if any pixel along a line matches a specific color

    Args:
        emulator: emulator instance
        x_1, y_1, x_2, y_2: line coordinates
        color: RGB color to check for

    Returns:
        bool: True if any pixel on line matches color
    """
    coordinates = get_line_coordinates(x_1, y_1, x_2, y_2)
    iar = np.asarray(emulator.screenshot())

    for coordinate in coordinates:
        pixel = iar[coordinate[1]][coordinate[0]]
        pixel = convert_pixel(pixel)

        if pixel_is_equal(color, pixel, tol=35):
            return True
    return False


def region_is_color(emulator, region: list, color: tuple[int, int, int]) -> bool:
    """Check if entire region matches a specific color (sampled every 2 pixels)

    Args:
        emulator: emulator instance
        region: [left, top, width, height] region to check
        color: RGB color to check for

    Returns:
        bool: True if entire region matches color
    """
    left, top, width, height = region
    iar = np.asarray(emulator.screenshot())

    for x_index in range(left, left + width, 2):
        for y_index in range(top, top + height, 2):
            pixel = iar[y_index][x_index]
            pixel = convert_pixel(pixel)
            if not pixel_is_equal(color, pixel, tol=35):
                return False

    return True


def all_pixels_are_equal(
    pixels_1: list,
    pixels_2: list,
    tol: float,
) -> bool:
    """Check if two lists of pixels are equal within tolerance

    Args:
        pixels_1: first list of pixels
        pixels_2: second list of pixels
        tol: color tolerance

    Returns:
        bool: True if all pixels match within tolerance
    """
    for pixel1, pixel2 in zip(pixels_1, pixels_2):
        if not pixel_is_equal(pixel1, pixel2, tol):
            return False
    return True


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_first_location(
    locations: list[list[int] | None],
    flip=False,
) -> list[int] | None:
    """Get the first valid location from a list of locations

    Args:
    ----
        locations: list of coordinate locations
        flip: whether to flip x,y coordinates

    Returns:
    -------
        list[int] | None: first valid location or None

    """
    return next(
        ([location[1], location[0]] if flip else location for location in locations if location is not None),
        None,
    )


def check_for_location(locations: list[list[int] | None]) -> bool:
    """Check if any location in the list is valid

    Args:
    ----
        locations: list of coordinate locations

    Returns:
    -------
        bool: True if any location is not None

    """
    return any(location is not None for location in locations)


def convert_pixel(bgr_pixel) -> list[int]:
    """Convert BGR pixel format to RGB

    Args:
        bgr_pixel: pixel in BGR format

    Returns:
        list[int]: pixel in RGB format [red, green, blue]
    """
    red = bgr_pixel[2]
    green = bgr_pixel[1]
    blue = bgr_pixel[0]
    return [red, green, blue]


def get_line_coordinates(x_1: int, y_1: int, x_2: int, y_2: int) -> list[tuple[int, int]]:
    """Get all pixel coordinates along a line using Bresenham's algorithm

    Args:
        x_1, y_1: start coordinates
        x_2, y_2: end coordinates

    Returns:
        list[tuple[int, int]]: list of (x, y) coordinates along the line
    """
    coordinates = []
    delta_x = abs(x_2 - x_1)
    delta_y = abs(y_2 - y_1)
    step_x = -1 if x_1 > x_2 else 1
    step_y = -1 if y_1 > y_2 else 1
    error = delta_x - delta_y

    while x_1 != x_2 or y_1 != y_2:
        coordinates.append((x_1, y_1))
        double_error = 2 * error
        if double_error > -delta_y:
            error -= delta_y
            x_1 += step_x
        if double_error < delta_x:
            error += delta_x
            y_1 += step_y

    coordinates.append((x_1, y_1))
    return coordinates
