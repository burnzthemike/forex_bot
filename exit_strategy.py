# exit_strategy.py

import datetime
from config import MAX_HOLD_TIME_MINUTES
from utils import log

def check_time_exit(entry_time: datetime.datetime, current_time: datetime.datetime = None) -> bool:
    """
    Returns True if the time elapsed since trade entry exceeds MAX_HOLD_TIME_MINUTES.
    
    Parameters:
        entry_time (datetime): The time when the trade was entered.
        current_time (datetime): Current time (optional, defaults to now).

    Returns:
        bool: True if trade should be closed due to time limit.
    """
    try:
        if current_time is None:
            current_time = datetime.datetime.utcnow()

        elapsed = current_time - entry_time
        should_exit = elapsed.total_seconds() > MAX_HOLD_TIME_MINUTES * 60

        if should_exit:
            log(f"[Exit] Trade held for {elapsed}, exceeding max hold of {MAX_HOLD_TIME_MINUTES} min. Exiting.")
        return should_exit

    except Exception as e:
        log(f"[Exit] Error checking time-based exit: {e}")
        return False
