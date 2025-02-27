import logging

class Logger:
    def __init__(self, log_file='app.log', log_level=logging.INFO):
        # 创建日志处理器
        self.file_handler = logging.FileHandler(log_file, encoding='utf-8')
        
        # 设置日志格式，包含文件名、行号和函数名等位置信息
        self.formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s - '
            'File: %(filename)s, Line: %(lineno)d, Func: %(funcName)s'
        )
        self.file_handler.setFormatter(self.formatter)

        # 获取日志记录器
        self.logger = logging.getLogger()
        
        # 设置日志级别
        self.logger.setLevel(log_level)
        
        # 添加处理器
        self.logger.addHandler(self.file_handler)

    def get_logger(self):
        return self.logger

    def set_level(self, level):
        """动态调整日志等级"""
        self.logger.setLevel(level)

    def add_console_handler(self):
        """添加控制台日志输出"""
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self.formatter)
        self.logger.addHandler(console_handler)