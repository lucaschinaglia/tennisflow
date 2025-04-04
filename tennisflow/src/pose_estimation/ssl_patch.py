# SSL verification patch
import ssl

# Disable SSL verification for the entire application
ssl._create_default_https_context = ssl._create_unverified_context
