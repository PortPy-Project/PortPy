"""This package options includes option modules: training options, test options, and basic options (used in both training and test)."""
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)