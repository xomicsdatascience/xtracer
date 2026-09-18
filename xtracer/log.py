import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path


class MyFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = time.time()

    def format(self, record):
        total_seconds = int(time.time() - self.start_time)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        record.elapsed_time = f"[{hours:02}:{minutes:02}:{seconds:02}]"
        return super().format(record)


class Logger:
    logger = logging.getLogger('xTracer')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    @classmethod
    def set_logger(cls, dir_out, run_name='xtracer', command=None, parameters=None):
        """Create one immutable, timestamped log for a command invocation."""
        logging._startTime = time.time()
        dir_out = Path(dir_out)
        dir_out.mkdir(parents=True, exist_ok=True)
        logtime = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_path = dir_out / f'{run_name}_{logtime}.log'

        fh = logging.FileHandler(log_path, mode='x', encoding='utf-8')
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = MyFormatter(fmt='%(elapsed_time)s: %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        for handler in list(cls.logger.handlers):
            cls.logger.removeHandler(handler)
            handler.close()
        cls.logger.addHandler(fh)
        cls.logger.addHandler(ch)

        cls.logger.info('xTracer run started')
        cls.logger.info('log_path: %s', log_path.resolve())
        cls.logger.info('python: %s', sys.version.replace('\n', ' '))
        from xtracer import __version__
        cls.logger.info('xtracer_version: %s', __version__)
        if command:
            cls.logger.info('command: %s', command)
        if parameters is not None:
            cls.logger.info('[effective_parameters]\n%s', json.dumps(
                parameters, indent=2, sort_keys=True, default=str,
            ))
        return log_path

    @classmethod
    def get_logger(cls):
        return cls.logger
