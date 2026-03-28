import traceback
from pathlib import Path

class Log:
    """
    @brief Utility class for standardised logging across the project.
    """

    @staticmethod
    def _get_file_name(file_path: str) -> str:
        """
        @brief Safely extracts the filename (without extension) from a path.
        @param file_path Full path to the file.
        @return Filename without extension.
        """
        try:
            return Path(file_path).stem
        except Exception:
            # Fallback to the raw string if Path conversion fails
            return str(file_path)

    @staticmethod
    def error(file_path: str, message: Exception):
        """
        @brief Logs an error message with the specific filename.
        @param file_path Full path to the file where the error occurred.
        @param message Exception message or object to log.
        """
        file_name = Log._get_file_name(file_path)
        tb = traceback.extract_tb(message.__traceback__)
        for frame in tb:
            print(f"[ {file_name} ] 🚨 Error line: {frame.lineno}; message: {message}")

    @staticmethod
    def warning(file_path: str, message: str):
        """
        @brief Logs a warning message with the specific filename.
        @param file_path Full path to the file issuing the warning.
        @param message Warning message to log.
        """
        file_name = Log._get_file_name(file_path)
        print(f"[ {file_name} ] ⚠️  Warning: {message}")

    @staticmethod
    def check(file_path: str, message: str):
        """
        @brief Logs a success/check message with the specific filename.
        @param file_path Full path to the file associated with the check.
        @param message Success message to log.
        """
        file_name = Log._get_file_name(file_path)
        print(f"[ {file_name} ] {message}   ✔️")

    @staticmethod
    def info(file_path: str, message: str):
        """
        @brief Logs an informational message with the specific filename.
        @param file_path Full path to the file issuing the info.
        @param message Informational message to log.
        """
        file_name = Log._get_file_name(file_path)
        print(f"[ {file_name} ] {message}")