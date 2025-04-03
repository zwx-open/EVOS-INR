from loguru import logger
import os
import datetime
import time
from util.misc import format_time, format_time2
from collections import defaultdict


class Logger:
    def __init__(self):
        self.inst = logger

        # timer
        self._default_task_name = "Task"
        self._task_dict = defaultdict(dict)
        self._show_timer_log = True


        # self.execution_time_list = []
        logger.level("Config", no=10, icon="&", color="<blue><bold>")
        logger.level("Time", no=10, icon="#", color="<yellow><bold>")
        logger.level("Var", no=10, icon="@", color="<green><bold>")

    def set_export(self, log_path, logfile_name=None):
        os.makedirs(log_path, exist_ok=True)

        if logfile_name is None:
            cur = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            logfile_name = f"{cur}.log"

        path_ = os.path.join(log_path, logfile_name)

        logger.add(
            path_,
            level="TRACE",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level.icon} | {message}",
            colorize=True,
        )
    def show_timer_log(self, bool):
        self._show_timer_log = bool

    def start_timer(self, task):
        if not self._show_timer_log : return

        cur_task_name = task if task is not None else self._default_task_name
        start_time = time.time()
        cur_task_dict = self._task_dict[cur_task_name]
        if cur_task_dict.get("execution") is None:
            self.inst.success(f"Start Timer | {cur_task_name} ...")
            cur_task_dict["start"] = start_time
            cur_task_dict["execution"] = 0
        else:
            # self.inst.success(f"Continue Timer | {cur_task_name} ...")
            cur_task_dict["start"] = start_time

        return start_time

    def pause_timer(self, task):
        if not self._show_timer_log : return
        pause_time = time.time()
        cur_task_name = task if task is not None else self._default_task_name
        cur_task_dict = self._task_dict[cur_task_name]
        start_time = cur_task_dict["start"]
        execution_time = pause_time - start_time
        cur_task_dict["execution"] += execution_time
        cur_task_dict["start"] = None

    def end_timer(self, task):
        if not self._show_timer_log : return
        end_time = time.time()
        cur_task_name = task if task is not None else self._default_task_name
        cur_task_dict = self._task_dict[cur_task_name]

        start_time = cur_task_dict["start"]
        execution_time = end_time - start_time if start_time is not None else 0
        execution_time += cur_task_dict["execution"]

        self.inst.success(f"End Timer | {cur_task_name} ...")
        self.inst.info(f"Total Execution Time for {cur_task_name}: {format_time(execution_time)}")

        cur_task_dict["execution"] = execution_time
        cur_task_dict["start"] = None

        return execution_time

    def end_all_timer(self):
        timer_dic = {}
        for task in self._task_dict:
            execution_time = self.end_timer(task)
            timer_dic[task] = format_time2(execution_time)
        return timer_dic


# export singleton
log = Logger()
