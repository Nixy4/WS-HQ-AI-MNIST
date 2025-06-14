import logging
import colorlog

loggerHandler = colorlog.StreamHandler()
loggerHandler.setFormatter(
    colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)s:%(name)s:%(message)s",
        log_colors={
            'DEBUG': 'white',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
    }
))
'''
设置日志处理器, 使用colorlog库来实现彩色日志输出
'''

logger = logging.getLogger("MNIST")
logger.addHandler(loggerHandler)
logger.setLevel(logging.DEBUG)

def logger_test():
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

log = logger # Alias for convenience
'''
重命名logger为log, 以便在其他脚本中使用时更简洁
'''

if __name__ == "__main__":
    logger_test()
    logger.info("Logger is set up and ready to use.")

'''
重命名本文件为 logger.py, 以便在作为模块在其他脚本中导入使用, 0-logger.py不是一个有效的模块名
'''