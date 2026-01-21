from rich.logging import RichHandler
from logging import Logger, getLogger, Formatter

class LoggerFactory:
    @staticmethod
    def create_logger(
        name: str, 
        level: str = "INFO",
        rich_tracebacks: bool = True,
        show_time: bool = True,
        show_path: bool = True
    ) -> Logger:
        """Logger factory method for creating loggers with RichHandler."""
        logger = getLogger(name)
        logger.setLevel(level)

        #Rich Handler with some modifications
        handler = RichHandler(
            rich_tracebacks=rich_tracebacks,
            show_time=show_time,
            show_path=show_path,
            markup=True  
        )
        handler.setLevel(level)

        if not logger.hasHandlers():
            logger.addHandler(handler)
        
        logger.propagate = False

        return logger

if __name__ == "__main__":
    my_logger = LoggerFactory.create_logger("my_logger", "DEBUG")
    my_logger.debug("This is a debug message")
    my_logger.info("This is an info message")
    my_logger.warning("This is a warning message")
    my_logger.error("This is an error message")
    my_logger.critical("This is a critical message")
    
    # You can also use rich markup
    my_logger.info("[bold green]Success![/bold green] Operation completed")