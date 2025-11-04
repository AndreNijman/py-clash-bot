"""Module to parse arguments from CLI"""

from argparse import ArgumentParser, Namespace


def arg_parser() -> Namespace:
    """Function to parse arguments

    Returns
    -------
        Namespace: populated namespace from arguments

    """
    parser = ArgumentParser(description="Run py-clash-bot from CLI")
    parser.add_argument(
        "--start",
        "-s",
        dest="start",
        action="store_true",
        help="Start the bot when the program opens",
    )
    parser.add_argument(
        "--capture-backend",
        dest="capture_backend",
        choices=["dxcam", "mss"],
        default="dxcam",
        help="Screen capture backend to use for emulator frames",
    )
    parser.add_argument(
        "--capture-title",
        dest="capture_title",
        default=None,
        help="Window title to target for desktop capture overrides",
    )
    parser.add_argument(
        "--capture-downscale",
        dest="capture_downscale",
        type=float,
        default=0.75,
        help="Downscale factor applied to captured frames (0 < factor ≤ 1)",
    )
    return parser.parse_args()
