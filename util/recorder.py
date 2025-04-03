from collections import OrderedDict
from collections import defaultdict
from prettytable import PrettyTable
from util.misc import AverageMeter
import math

class Recorder(object):
    def __init__(self):
        self.dic= OrderedDict()
        self.average_meter_dic = OrderedDict() 
        self.tables = []
        self.summary_recodrder = defaultdict(dict)
        self._max_col = 8
    
    def add_main_table(self):
        table = None
        samples = list(self.dic.keys())
        
        epochs = list(self.dic[samples[0]].keys())
        
        for epoch in epochs:
            attrs = list(self.dic[samples[0]][epoch].keys())

            table_num = math.ceil(len(attrs) / self._max_col)
            for attr in attrs:
                self.average_meter_dic[attr] = AverageMeter(attr)

            for i in range(table_num):
                cur_attrs = attrs[i*self._max_col:min(len(attrs),(i+1)*self._max_col)]

                headers = [f"epoch_{epoch}"] + cur_attrs
                table = PrettyTable(headers)
                table.float_format = ".8"
            
                for sample in samples:
                    cur_record = self.dic[sample][epoch]
                    results = [cur_record[attr] for attr in cur_attrs]
                    table.add_row([sample] + results)
                    for attr in cur_attrs:
                        self.average_meter_dic[attr].update(cur_record[attr])
                    
                table.add_row(["mean"] + [self.average_meter_dic[attr].avg for attr in cur_attrs])
                
                # record for summary table
                for attr in cur_attrs:
                    self.summary_recodrder[epoch][attr] = self.average_meter_dic[attr].avg
                
                self.tables.append(table)
        print(table)
        return table
    
    def add_summary_table(self, epochs):
        # diy: final epoch table for paper writting
        epochs = [1000, 2000, 5000]
        # epochs = [100,500,1000]
        attrs = ["psnr", "ssim", "lpips"]
        headers = [f"summary"] + [f"{epoch}_{attr}" for epoch in epochs for attr in attrs]
        table = PrettyTable(headers)
        table.float_format = ".4"
        to_add = ["mean"]
        for epoch in epochs:
            if self.summary_recodrder.get(epoch) is None: continue
            to_add += [self.summary_recodrder[epoch][attr] for attr in attrs ]
        table.add_row(to_add)
        self.tables.append(table)

    def add_time_table(self):
        time_dic = self.dic.get("time", None)
        if time_dic is None: return 
        tasks = list(time_dic.keys())
        headers = [f"time"] + tasks
        table = PrettyTable(headers)
        row = ["time"]+[time_dic[task] for task in tasks]
        table.add_row(row)
        self.tables.append(table)
        
    
    def dump_table(self, save_path: str, type: str = "md"):
        for table in self.tables:
            with open(save_path, "a") as file_handler:
                if type == "md":
                    file_handler.writelines(table.get_string())
                elif type == "csv":
                    file_handler.writelines(table.get_csv_string())
                else:
                    raise NotImplementedError
                file_handler.write("\n")


# export singleton 
recorder = Recorder()