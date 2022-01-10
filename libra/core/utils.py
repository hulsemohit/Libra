import datetime

import sys

LOGFILE = sys.stderr
LOGLEVEL = 3


def debug(message: str):
    if LOGLEVEL >= 3:
        log("[D] " + message)


def info(message: str):
    if LOGLEVEL >= 2:
        log("[I] " + message)


def warn(message: str):
    if LOGLEVEL >= 1:
        log("[W] " + message)


def fail(message: str):
    if LOGLEVEL >= 0:
        log("[F] " + message)
    raise Exception(message)


def clear():
    try:
        LOGFILE.truncate(0)
    except OSError:
        # This should only occur if using stdout/stderr or an invalid file
        pass


def log(message: str):
    LOGFILE.write(f"[{str(datetime.datetime.now())}] <core> " + message + "\n")
