import os
import datetime

class SlideAccessTime:
    def __init__(self):
        pass
        self.current_time = None

    def set_current_time(self):
        self.current_time = datetime.datetime.now()

    def set_last_time_from_file(self, filename=None):
        with open(filename, "r") as ff:
            data = ff.read()
            data = data.split(' ')
            data_list = [int(x) for x in data]
            self.current_time = datetime.datetime(year=data_list[0], month=data_list[1],
                                                  day = data_list[2], hour=data_list[3],
                                                  minute=data_list[4])

    def save_current_time(self, filename):
        with open(filename,"w") as ff:
            self.current_time = datetime.datetime.now()
            cur_str = f'{self.current_time.year} {self.current_time.month} {self.current_time.day} {self.current_time.hour} {self.current_time.minute}'
            ff.write(cur_str)

    def filter_files(self, files_list:list=[]):
        ret_files = []
        for fn in files_list:
            c_timestamp = os.path.getctime(fn)
            date_obj = datetime.datetime.fromtimestamp(c_timestamp)
            if self.current_time is None or date_obj > self.current_time:
                ret_files.append(fn)
        return ret_files
