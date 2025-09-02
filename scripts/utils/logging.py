# utils/logging.py
from __future__ import annotations
import os, sys, datetime, atexit

from pathlib import Path
import csv, json
from typing import Dict, Any, Optional


class _Tee:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass
        self.flush()
    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

def setup_logging(logdir: str, filename: str = "train.log") -> None:
    os.makedirs(logdir, exist_ok=True)
    log_path = os.path.join(logdir, filename)

    # line-buffered so prints appear promptly
    fh = open(log_path, "w", encoding="utf-8", buffering=1)
    orig_out, orig_err = sys.stdout, sys.stderr

    tee = _Tee(sys.__stdout__ or orig_out, fh)
    sys.stdout = tee
    sys.stderr = tee

    def _cleanup():
        # restore first, so late prints go only to console (not a closed file)
        sys.stdout = orig_out
        sys.stderr = orig_err
        try: fh.flush()
        except Exception: pass
        try: fh.close()
        except Exception: pass

    atexit.register(_cleanup)
    print(f"# Logging to {log_path}   ({datetime.datetime.now().isoformat(timespec='seconds')})")

class CSVHistoryLogger:
    """
    Origin: Python stdlib (csv, json).
    Purpose: Append (epoch, split) metrics to a CSV; save run summary to JSON.

    Fields written to CSV (stable schema):
    epoch,split,loss,acc,macro_f1,lr,wall_time
    """
    def __init__(self, logdir: Path, filename: str = "history.csv") -> None:
        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.path = self.logdir / filename
        self._initialized = self.path.exists()
        self._fieldnames = ["epoch","split","loss","acc","macro_f1","lr","wall_time"]

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

