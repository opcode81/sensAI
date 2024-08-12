import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

print(os.environ["PYTHONPATH"] + os.pathsep + os.path.abspath(os.path.join(SCRIPT_DIR, "..", "src")))
