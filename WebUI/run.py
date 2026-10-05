#!/usr/bin/env python3
"""
Launcher script for the 6D Detection UI system.

This script uses EnvManager to locate the Data_Generation virtual environment
and executes the original startup script under that interpreter.
"""

import os
import sys
from pathlib import Path

from tomo_system import get_env_manager

SCRIPT_PATH = Path(__file__).resolve()
RUN_APP_PATH = SCRIPT_PATH.with_name("run_app.py")
ENV_NAME = "Data_Generation"
REQUIREMENTS_PATH = "../requirements.txt"


def bootstrap_target_environment():
    """Execute run_app.py under Data_Generation virtual environment interpreter."""
    env_manager = get_env_manager()
    target_python = env_manager.get(ENV_NAME, REQUIREMENTS_PATH)

    if not RUN_APP_PATH.exists():
        raise FileNotFoundError(f"run_app.py not found: {RUN_APP_PATH}")

    print(f"[*] Launching with virtual environment '{ENV_NAME}'")
    print(f"target python: {target_python}")
    os.environ['DATA_GEN_PYTHON'] = str(target_python)
    os.execv(
        str(target_python),
        [str(target_python), str(RUN_APP_PATH), *sys.argv[1:]],
    )

bootstrap_target_environment()
