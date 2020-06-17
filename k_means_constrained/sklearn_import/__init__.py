import os


def get_config():
    """Retrieve current values for configuration set by :func:`set_config`

    Returns
    -------
    config : dict
        Keys are parameter names that can be passed to :func:`set_config`.
    """
    return {'assume_finite': _ASSUME_FINITE}


__version__ = '0.19.2'
_ASSUME_FINITE = bool(os.environ.get('SKLEARN_ASSUME_FINITE', False))