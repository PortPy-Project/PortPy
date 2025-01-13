"""This package includes a miscellaneous collection of useful helper functions."""
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)
