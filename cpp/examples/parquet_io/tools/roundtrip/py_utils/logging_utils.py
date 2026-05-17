"""
Logging utilities with colored console output and file logging.

Provides a custom logger that writes timestamped, colored messages to
both console and an optional log file.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, TextIO


class Colors:
    """ANSI color codes for terminal output."""
    
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    CYAN = "\033[0;36m"
    MAGENTA = "\033[0;35m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    
    @classmethod
    def disable(cls) -> None:
        """Disable colors (for non-TTY output)."""
        cls.RED = ""
        cls.GREEN = ""
        cls.YELLOW = ""
        cls.BLUE = ""
        cls.CYAN = ""
        cls.MAGENTA = ""
        cls.BOLD = ""
        cls.RESET = ""


class ColoredLogger:
    """
    Logger that writes colored output to console and plain text to file.
    
    Supports different log levels with appropriate colors:
    - INFO: Blue
    - PASS: Green
    - WARN: Yellow
    - FAIL: Red
    - HEADER: Cyan + Bold
    """
    
    def __init__(
        self,
        name: str = "parquet_verifier",
        log_file: Optional[Path] = None,
        console_output: bool = True,
    ):
        """
        Initialize the colored logger.
        
        Args:
            name: Logger name for identification.
            log_file: Optional path to log file.
            console_output: Whether to print to console.
        """
        self.name = name
        self.console_output = console_output
        self.log_file: Optional[TextIO] = None
        self.log_file_path: Optional[Path] = None
        self.start_time = datetime.now()
        
        # Disable colors if stdout is not a TTY
        if not sys.stdout.isatty():
            Colors.disable()
        
        if log_file:
            self.enable_file_logging(log_file)
    
    def enable_file_logging(self, filepath: Path) -> None:
        """
        Enable logging to a file.
        
        Args:
            filepath: Path to the log file.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(filepath, "w", encoding="utf-8")
        self.log_file_path = filepath
        
        # Write header to log file
        self._write_log_header()
    
    def _write_log_header(self) -> None:
        """Write header information to log file."""
        if self.log_file:
            import socket
            header = f"""{'=' * 80}
Parquet Compression Roundtrip Verification Log
{'=' * 80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Host: {socket.gethostname()}
{'=' * 80}

"""
            self.log_file.write(header)
            self.log_file.flush()
    
    def close(self) -> None:
        """Close the log file if open."""
        if self.log_file:
            self.log_file.close()
            self.log_file = None
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        return datetime.now().strftime("%H:%M:%S")
    
    def _write_to_file(self, message: str) -> None:
        """Write message to log file without colors."""
        if self.log_file:
            self.log_file.write(message + "\n")
            self.log_file.flush()
    
    def _print_to_console(self, message: str) -> None:
        """Print message to console."""
        if self.console_output:
            print(message)
    
    def log(self, message: str, level: str = "INFO", console: bool = True) -> None:
        """
        Log a message with timestamp and level.
        
        Args:
            message: The message to log.
            level: Log level (INFO, PASS, WARN, FAIL).
            console: Whether to print to console.
        """
        timestamp = self._get_timestamp()
        
        # Color mapping
        color_map = {
            "INFO": Colors.BLUE,
            "PASS": Colors.GREEN,
            "WARN": Colors.YELLOW,
            "FAIL": Colors.RED,
        }
        color = color_map.get(level, Colors.RESET)
        
        # Formatted messages
        plain_msg = f"[{timestamp}] [{level}] {message}"
        colored_msg = f"{color}{plain_msg}{Colors.RESET}"
        
        self._write_to_file(plain_msg)
        if console:
            self._print_to_console(colored_msg)
    
    def info(self, message: str, console: bool = True) -> None:
        """Log an INFO level message."""
        self.log(message, "INFO", console)
    
    def success(self, message: str, console: bool = True) -> None:
        """Log a PASS level message."""
        self.log(message, "PASS", console)
    
    def warning(self, message: str, console: bool = True) -> None:
        """Log a WARN level message."""
        self.log(message, "WARN", console)
    
    def error(self, message: str, console: bool = True) -> None:
        """Log a FAIL level message."""
        self.log(message, "FAIL", console)
    
    def header(self, message: str, console: bool = True) -> None:
        """
        Log a header message (bold cyan).
        
        Args:
            message: The header text.
            console: Whether to print to console.
        """
        self._write_to_file("")
        self._write_to_file(message)
        self._write_to_file("-" * len(message))
        
        if console and self.console_output:
            print(f"\n{Colors.BOLD}{Colors.CYAN}{message}{Colors.RESET}")
    
    def raw(self, message: str, console: bool = True) -> None:
        """
        Log a raw message without timestamp or level.
        
        Args:
            message: The message to log.
            console: Whether to print to console.
        """
        self._write_to_file(message)
        if console:
            self._print_to_console(message)
    
    def metric(
        self,
        label: str,
        value: str,
        unit: str = "",
        console: bool = True,
    ) -> None:
        """
        Log a metric with aligned formatting.
        
        Args:
            label: Metric label.
            value: Metric value.
            unit: Optional unit string.
            console: Whether to print to console.
        """
        formatted = f"  {label:<25}: {value} {unit}".rstrip()
        self._write_to_file(formatted)
        if console:
            self._print_to_console(formatted)
    
    def table_row(
        self,
        columns: list,
        widths: list,
        color: Optional[str] = None,
        console: bool = True,
    ) -> None:
        """
        Log a formatted table row.
        
        Args:
            columns: List of column values.
            widths: List of column widths.
            color: Optional color for the row.
            console: Whether to print to console.
        """
        formatted_cols = []
        for col, width in zip(columns, widths):
            formatted_cols.append(f"{col:<{width}}")
        
        row = " | ".join(formatted_cols)
        plain_row = f"  {row}"
        
        self._write_to_file(plain_row)
        if console and self.console_output:
            if color:
                print(f"  {color}{row}{Colors.RESET}")
            else:
                print(plain_row)
    
    def separator(self, char: str = "-", width: int = 79, console: bool = True) -> None:
        """Log a separator line."""
        line = f"  {char * width}"
        self._write_to_file(line)
        if console:
            self._print_to_console(line)
    
    def __enter__(self) -> "ColoredLogger":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - close log file."""
        self.close()