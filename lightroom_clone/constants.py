import os

# Supported file extensions
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".gif"]

# Common RAW formats
RAW_EXTENSIONS = [
    ".cr2",
    ".cr3",
    ".nef",
    ".nrw",
    ".arw",
    ".orf",
    ".rw2",
    ".raf",
    ".dng",
    ".pef",
    ".srw",
    ".rwl",
    ".3fr",
    ".erf",
    ".kdc",
    ".mrw",
    ".sr2",
    ".srf",
]

# Combined extensions used throughout the app
IMAGE_EXTENSIONS = IMAGE_EXTENSIONS + RAW_EXTENSIONS

LRC_VERSION = "1.0"

# Project root defaults to the directory containing the entry script
PROJECT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
