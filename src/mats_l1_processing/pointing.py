"""
Author Donal Murtagh
"""
import numpy as np


def add_channel_quaternion(CCDitem):
    """ Add channel quaternion to CCDimage This quaternion converts
    from channel coordinated to OHB body coordinates

    Args:
        CCDitem

    """
    CCDitem['qprime'] = CCDitem["CCDunit"].get_channel_quaternion()
    return
