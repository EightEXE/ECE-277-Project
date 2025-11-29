import os

# Supported file extensions
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".gif"]

# Common RAW formats (dedicated list to avoid Qt mis-loading)
RAW_EXTENSIONS = {
    ".cr2", ".cr3", ".nef", ".nrw", ".arw", ".orf",
    ".raf", ".rw2", ".dng", ".sr2", ".srw", ".pef",
}

# Combined extensions used throughout the app
IMAGE_EXTENSIONS = IMAGE_EXTENSIONS + list(RAW_EXTENSIONS)

LRC_VERSION = "1.0"

PROJECT_EXTENSION = ".gsp"
LEGACY_PROJECT_EXTENSION = ".lrc"

# Project root defaults to the directory containing the entry script
PROJECT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
