import os
import hubpersistence

STATIC_FOLDER = os.path.join(os.path.dirname(os.path.dirname(gpa.__file__)),
                             "static")
DATA_FOLDER = os.path.join(STATIC_FOLDER, "data")