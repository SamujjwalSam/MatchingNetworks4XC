# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6
"""
__synopsis__    : logging to console and file.

__description__ : Logger class for the project.
__project__     : Extreme Classification
__author__      : Samujjwal Ghosh
__version__     :
__date__        : June 2018
__copyright__   : "Copyright (c) 2019"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root
                  directory of this source tree.

__classes__     : ColoredFormatter

__variables__   :

__methods__     :
"""

import logging
from json import dumps
from os import makedirs
from os.path import join,exists
import sys
from copy import copy
from datetime import datetime
from logging import FileHandler
from logging import Formatter

LOG_FILE = datetime.now().strftime('%Y%m%d_%H%M%S')

MAPPING = {
    'DEBUG'   :37,  # white
    'INFO'    :36,  # cyan
    'WARNING' :33,  # yellow
    'ERROR'   :31,  # red
    'CRITICAL':41,  # white on red bg
}

PREFIX = '\033['
SUFFIX = '\033[0m'


class ColoredFormatter(Formatter):
    """
    Class to add color support for logging.
    """

    def __init__(self,patern):
        Formatter.__init__(self,patern)

    def format(self,record: logging.LogRecord) -> str:
        colored_record = copy(record)
        levelname = colored_record.levelname
        seq = MAPPING.get(levelname,37)  # default white
        colored_levelname = '{0}{1}m{2}{3}'.format(PREFIX,seq,levelname,SUFFIX)
        colored_record.levelname = colored_levelname
        return Formatter.format(self,colored_record)


def create_logger(logger_name: str = 'root',
                  log_filename: str = LOG_FILE,
                  file_path: str = 'logs',
                  file_level: int = logging.DEBUG,
                  file_format: str = "%(asctime)s [%(levelname)s %(funcName)s] (%(module)s:%(lineno)d) %(message)s",
                  console_level: int = logging.DEBUG,
                  console_format: str = "[%(levelname)s] [%(module)s (%(lineno)d): %(funcName)s] %(message)s",
                  color: bool = True,
                  ) -> logging.Logger:
    """
    Creates logger with console and file printing.

    :param color:
    :param logger_name:
    :param log_filename:
    :param file_path:
    :param file_format:
    :param file_level:
    :param console_format:
    :param console_level:
    :return:
    """
    if not exists(file_path):
        makedirs(file_path)
    logger = logging.getLogger(logger_name)
    logger.setLevel(file_level)
    file_logger = FileHandler(join(file_path,log_filename + '.log'))
    file_logger.setLevel(file_level)
    file_logger.setFormatter(Formatter(file_format))
    logger.addHandler(file_logger)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(console_level)
    if color:
        console.setFormatter(ColoredFormatter(console_format))
    else:
        console.setFormatter(Formatter(console_format))
    logger.addHandler(console)
    return logger


def print_dict(data,count=5):
    """
    Prints the key and values of a Python dict.

    :param data:
    :param count:
    """
    i = 0
    for k,v in data.items():
        logger.debug("{} : {}".format(k,v))
        i += 1
        if i >= count:
            break


def print_json(json_data,s="",indent=4,sort_keys=True):
    """
    Pretty prints json data.

    :param sort_keys:
    :param indent:
    :param s:
    :param json_data:
    """
    logger.info("[{}] : {}".format(s,dumps(json_data,indent=indent,sort_keys=sort_keys)))


def get_date_time_tag(caller=False):
    from datetime import datetime

    date_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = str(date_time)
    if caller:
        tag = caller + "_" + str(date_time)
    return tag


def _test_colored_logger():
    # Create top level logger
    log = logging.getLogger("main")

    # Add console handler using our custom ColoredFormatter
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    cf = ColoredFormatter("[%(name)s][%(levelname)s]  %(message)s (%(filename)s:%(lineno)d)")
    ch.setFormatter(cf)
    log.addHandler(ch)

    # Add file handler
    fh = logging.FileHandler('app.log')
    fh.setLevel(logging.DEBUG)
    ff = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(ff)
    log.addHandler(fh)

    # Set log level
    log.setLevel(logging.DEBUG)

    # Log some stuff
    log.debug("app has started")
    log.info("Logging to 'app.log' in the script dir")
    log.warning("This is my last warning, take heed")
    log.error("This is an error")
    log.critical("He's dead, Jim")


logger = create_logger(logger_name='MNXC.tools',log_filename='MNXC_' + LOG_FILE)
# DXMLclass_logger    = create_logger(logger_name='DXML.class')
# util_logger         = create_logger(logger_name='DXML.util')


if __name__ == "__main__":
    _test_colored_logger()
