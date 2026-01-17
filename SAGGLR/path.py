# Define the DIR to the data, model, logs, results, and colors
#
import os

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(ROOT_PATH, "data_preprocess/data")
FEATURE_ATTRI_PATH = os.path.join(ROOT_PATH, "feature_attribution")
MODEL_PATH = os.path.join(ROOT_PATH, "model")
LOG_PATH = os.path.join(ROOT_PATH, "logs")
COLOR_PATH = os.path.join(ROOT_PATH, "colors")
CV_PATH = os.path.join(ROOT_PATH, "cross_validation")
CV_PATH_TEST = os.path.join(ROOT_PATH, "cross_validation_test")
