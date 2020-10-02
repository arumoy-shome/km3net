import os
import sys

ROOTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATADIR = os.path.join(ROOTDIR, 'data')
MODELDIR = os.path.join(ROOTDIR, 'models')

__POSITIVE_RESPONSES = ['y', 'yes']
def yes(response):
    """
    In
    --
    response -> Str, response of user from input().

    Out
    ---
    Bool, True if response was affirmative else False.
    """
    return response.lower() in __POSITIVE_RESPONSES
