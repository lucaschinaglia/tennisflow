# Import SSL patch first to ensure SSL verification is disabled
try:
    from . import ssl_patch
except ImportError:
    pass

