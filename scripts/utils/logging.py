# utils/logging.py
import os, sys, datetime, atexit
from __future__ import annotations
from pathlib import Path
import csv, json
from typing import Dict, Any, Optional


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


class CSVHistoryLogger:
    """
    Origin: Python stdlib (csv, json).
    Purpose: Append (epoch, split) metrics to a CSV; save run summary to JSON.

    Fields written to CSV (stable schema):
    epoch,split,loss,acc,macro_f1,ua,lr,wall_time
    """
    def __init__(self, logdir: Path, filename: str = "history.csv") -> None:
        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.path = self.logdir / filename
        self._initialized = self.path.exists()
        self._fieldnames = ["epoch","split","loss","acc","macro_f1","ua","lr","wall_time"]

    def log(self, **row: Any) -> None:
        # ensure known fields; fill missing with ""
        record = {k: row.get(k, "") for k in self._fieldnames}
        with self.path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._fieldnames)
            if not self._initialized:
                w.writeheader()
                self._initialized = True
            w.writerow(record)

    def save_summary(self, *, best_dev_epoch: int, best_dev_macro_f1: float,
                     early_stop_epoch: Optional[int] = None) -> None:
        summary = {
            "best_dev_epoch": int(best_dev_epoch),
            "best_dev_macro_f1": float(best_dev_macro_f1),
            "early_stop_epoch": (None if early_stop_epoch is None else int(early_stop_epoch)),
        }
        (self.logdir / "metrics.json").write_text(json.dumps(summary, indent=2))
