# utils/logging_utils.py
import os, sys, datetime, atexit

def logging(logdir: str, filename: str = "train.log") -> None:
    """
    Redirect stdout *and* stderr so that everything printed during training
    is also saved in `logdir/filename`.

    Parameters
    ----------
    logdir : str
        Path to the run-specific folder (already made with os.makedirs).
    filename : str, default "train.log"
        Name of the text file that will hold the console output.
    """
    os.makedirs(logdir, exist_ok=True)
    log_path = os.path.join(logdir, filename)

    # -- minimal "tee" implementation -----------------------------------------
    class _Tee:
        def __init__(self, *streams):
            self._streams = streams
        def write(self, data):
            for s in self._streams:
                s.write(data)
                s.flush()
        def flush(self):
            for s in self._streams:
                s.flush()

    # open the file in **text** mode and redirect the std-streams
    _file_handle = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, _file_handle)   # keep live console echo
    sys.stderr = _Tee(sys.__stderr__, _file_handle)

    # make sure the file is closed when the program ends
    atexit.register(_file_handle.close)

    # first line so you see it both on screen and in the file
    print(f"# Logging to {log_path}   ({datetime.datetime.now().isoformat(timespec='seconds')})")
