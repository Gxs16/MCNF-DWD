import io
import logging
import logging.config

log_config_dict = {
    'version': 1.0,
    'disable_existing_loggers': False,
    'formatters': {
        'detail': {
            'format': '%(asctime)s %(levelname)s %(processName)s %(name)s-%(lineno)s: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
        'simple': {
            'format': '%(asctime)s-%(levelname)s-%(processName)s-%(module)s: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
        'only_time': {
            'format': '%(asctime)s: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
        'only_msg': {
            'format': '%(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
    },
    'handlers': {
        'to_console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'simple',
        },
        'to_console_only_msg': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'only_msg',
        },
        'to_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'maxBytes': 1024 * 1024 * 5,
            'backupCount': 10,
            'filename': 'log.log',
            'level': 'DEBUG',
            'formatter': 'detail',
            'encoding': 'utf-8',
        },
        'to_file_only_time': {
            'class': 'logging.handlers.RotatingFileHandler',
            'maxBytes': 1024 * 1024 * 5,
            'backupCount': 10,
            'filename': 'log.log',
            'level': 'DEBUG',
            'formatter': 'only_time',
            'encoding': 'utf-8',
        }
    },
    'loggers': {
        'cplex-solver': {
            'handlers': ['to_console_only_msg', 'to_file_only_time'],
            'level': 'DEBUG',
            'propagate': False,
        },
    },
    'root': {
        'handlers': ['to_console', 'to_file'],
        'level': 'INFO',
    },
}


class LoggerStream(io.IOBase):
    def __init__(self):
        super().__init__()

    def write(self, __s: str) -> int:
        logging.getLogger('cplex-solver').info(__s[:-1])
        return len(__s)