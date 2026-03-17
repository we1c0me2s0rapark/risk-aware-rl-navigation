import sys
import select

__all__ = ["is_q_pressed"] # Only this function is intended for external use

def is_key_pressed(target_key: str) -> bool:
    """
    @brief Internal helper: determines whether a specific key has been pressed.

    Uses non-blocking standard input to detect a single key press without
    requiring the Enter key.

    @note This function is considered private and is not intended for import
    outside this module.

    @param target_key The key to check for (case-insensitive).
    @return True if the specified key was pressed; otherwise False.
    """
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    if dr:
        ch = sys.stdin.read(1)
        return ch.lower() == target_key.lower()
    return False

def is_q_pressed() -> bool:
    """
    @brief Public convenience function for detecting the 'q' key.

    Wraps the internal helper to provide a clean interface for main scripts.

    @return True if the 'q' key was pressed; otherwise False.
    """
    return is_key_pressed('q')
