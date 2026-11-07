from enum import Enum


class Connectivity(Enum):
    """
    A class to represent connectivity.

    Parameters
    ----------
    FOUR: int
        FOUR is 4-connectivity.
    EIGHT: int
        EIGHT is 8-connectivity.
    """

    FOUR = 4
    EIGHT = 8
