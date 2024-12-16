import logging


class HttpFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "HTTP Request:" not in record.getMessage()


def start_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    logger = logging.getLogger()

    logger.addFilter(lambda record: not record.getMessage().startswith("HTTP Request:"))

    for logger_name in logging.root.manager.loggerDict.keys():
        logger = logging.getLogger(logger_name)
        logger.addFilter(HttpFilter())


start_logging()
