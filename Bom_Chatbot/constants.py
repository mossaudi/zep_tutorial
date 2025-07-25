# constants.py
"""Application constants."""

# API Configuration
DEFAULT_PAGE_SIZE = 5
MAX_SEARCH_RESULTS = 5
MAX_DESCRIPTION_LENGTH = 100
MAX_FEATURES_LENGTH = 50

# Progress Messages
PROGRESS_SYMBOLS = {
    'progress': '🔄',
    'completed': '✅',
    'error': '❌'
}

# Component Data Keys
COMPONENT_KEYS = {
    'NAME': ['name', 'component_name'],
    'PART_NUMBER': ['part_number', 'se_part_number', 'mpn'],
    'MANUFACTURER': ['manufacturer', 'se_manufacturer'],
    'DESCRIPTION': ['description', 'se_description']
}

# BOM Column Defaults
DEFAULT_BOM_COLUMNS = [
    "cpn", "mpn", "manufacturer", "description",
    "quantity", "uploadedcomments", "uploadedlifecycle"
]

# HTTP Status Codes
HTTP_OK = 200
HTTP_UNAUTHORIZED = 401
HTTP_NOT_FOUND = 404
HTTP_SERVER_ERROR = 500