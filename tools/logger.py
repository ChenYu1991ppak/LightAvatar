import logging

logging.basicConfig(
    level=logging.INFO,  # 修改日志级别
    format='%(asctime)s %(name)s [%(pathname)s line:%(lineno)d] %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',  # 日期时间格式
    filename='log.txt',  # 日志文件名
    filemode='w'  # 日志模式，如：w：写、a+：追加写 等
)


logger = logging.getLogger()
