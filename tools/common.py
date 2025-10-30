import os
import subprocess
import yaml
import re
import numpy as np
from enum import Enum
import logging
from logging import handlers
import glob


DEFAULT_CONFIG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config/base.yaml'))
SOURCE_IMAGE_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/source_images'))

CODE = {
    'SUCC': 200,  # 成功响应
    'RETURN': 210,  # 返回消息
    'RETURN_STREAM': 211,  # 流式返回
    'CFAIL': 400,
    'CFAIL_IDENT': 401,
    'CFAIL_PARAM': 402,
    'SFAIL': 500,  # 服务错误
    'SFAIL_ALGO': 501,
    'SFAIL_MISS_RESOURCE': 502,  # 资源缺失
    'SFAIL_EXCEED': 510,  # 超出连接数限制
}
def wscode(code: int) -> int:
    return 3000 + code
# 交互信号
SIGNAL_STOP_TALKING = '_stop_talking'
SIGNAL_CHECK_STATUS = '_check_status'


class TalkStatus(Enum):
    waiting = 'WAITING'
    speaking = 'SPEAKING'
    other = 'OTHER'


def load_yaml(path):
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))

    with open(path, 'r') as f:
        cfg_dict = yaml.load(f, Loader=loader)
    return cfg_dict


def load_model_paths(cfg_path):
    def update_path(root, path):
        if isinstance(path, str):
            return os.path.join(root, path)
        elif isinstance(path, dict):
            return {k: update_path(root, v) for k, v in path.items()}
        else:
            return path
    cfg = load_yaml(cfg_path)
    data_root = cfg['data_root']
    if not os.path.isabs(data_root):
        data_root = os.path.join(os.path.dirname(cfg_path), '../', data_root)
        data_root = os.path.abspath(data_root)
    download_home = cfg['download_home']

    save_paths = update_path(data_root, cfg['pre_trained_model'])
    download_urls = update_path(download_home, cfg['pre_trained_model'])
    return download_urls, save_paths


def load_emb2exp_map(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    return data['embs'], data['exps'], data['emb2exp']


def create_logger(name, level=logging.DEBUG, when='D', backCount=30, log_root=None,
                fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
    log_path = os.path.join(log_root, name + '.log') if log_root is not None else None
    logger = logging.getLogger(log_path)
    logger.setLevel(level)
    format_str = logging.Formatter(fmt)

    if not logger.handlers:
        sh = logging.StreamHandler()
        sh.setFormatter(format_str)
        sh.setLevel(level)

        if log_path is not None:
            subprocess.run(f"mkdir -p {log_root}", shell=True, check=True)
            th = handlers.TimedRotatingFileHandler(
                filename=log_path,
                when=when,
                backupCount=backCount,
                encoding='utf-8',
                delay=True
            )
            th.setFormatter(format_str)
            th.setLevel(level)

        logger.addHandler(sh)
        logger.addHandler(th)

    logger.propagate = False
    return logger


def get_virtual_human_source_folder(virtual_human_id: str) -> str:
    dataset_dir = glob.glob(os.path.join(SOURCE_IMAGE_FOLDER, virtual_human_id + '_*_video'))[0]
    if not os.path.exists(dataset_dir):
        return os.path.join(SOURCE_IMAGE_FOLDER, virtual_human_id + f'_zh_video')
    else:
        return dataset_dir