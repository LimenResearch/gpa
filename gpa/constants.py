import os
import gpa

MAIN_FOLDER = os.path.dirname(os.path.dirname(gpa.__file__))
STATIC_FOLDER = os.path.join(MAIN_FOLDER, "static")
DATA_FOLDER = os.path.join(STATIC_FOLDER, "data")