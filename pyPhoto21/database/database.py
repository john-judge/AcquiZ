import numpy as np
from matplotlib.path import Path

from pyPhoto21.analysis.core import AnalysisCore
from pyPhoto21.viewers.trace import Trace
from pyPhoto21.database.file import File
from pyPhoto21.database.metadata import Metadata

import os


class Database(File):

    def __init__(self):
        super().__init__()
        self.meta = Metadata()
        self.extension = '.npy'
        self.memmap_file = None
        self.open_filename = None

    def get_current_filename(self, no_path=False):
        fn = ''
        if self.open_filename is not None:
            return fn + self.open_filename
        if self.override_filename is not None:
            return fn + self.override_filename

        path = None
        if not no_path:
            path = self.get_save_dir()
        return self.get_filename(self.meta.current_slice,
                                 self.meta.current_location,
                                 self.meta.current_record,
                                 extension=self.extension,
                                 path=path)

    # https://numpy.org/doc/stable/reference/generated/numpy.memmap.html#numpy.memmap
    def load_mmap_file(self, mode='w+'):  # w+ allows create/overwrite. Set mode=None to auto-determine
        fn = self.get_current_filename()
        if mode is None:
            mode = 'w+'
            if self.file_exists(fn):
                print("r+")
                mode = 'r+'
        self.memmap_file = np.memmap(fn,
                                     dtype=np.uint16,
                                     mode=mode,
                                     shape=(self.meta.num_trials,
                                            2,
                                            self.meta.num_pts,
                                            self.meta.width,
                                            self.meta.height))
        self.open_filename = fn

    def clear_or_resize_mmap_file(self):
        # should avoid overwriting data by renaming current file
        if self.meta.auto_save_enabled and self.file_exists(self.get_current_filename(no_path=True)):
            print("Archiving file: ", self.get_current_filename())
            i = 0
            new_name = self.get_current_filename(no_path=True) + ".archive_" + self.pad_zero(i, dst_len=2)
            while self.file_exists(new_name) or i > 100:
                i += 1
                new_name = self.get_current_filename(no_path=True) + ".archive_" + self.pad_zero(i, dst_len=2)
            try:
                new_name = self.get_current_filename() + ".archive_" + self.pad_zero(i, dst_len=2)
                os.rename(self.get_current_filename(), new_name)
            except Exception as e:
                print("Could not archive current data file: ",
                      self.get_current_filename(),
                      "Delete or move this file, or choose a new file to proceed with array resize.",
                      "\nActual exception:")
                print(e)
                return
        self.load_mmap_file()

    def load_trial_data_raw(self, trial):
        return self.memmap_file[trial, 0, :, :, :]

    def load_trial_data_processed(self, trial):
        return self.memmap_file[trial, 1, :, :, :]

    def load_data_raw(self):
        return self.memmap_file[:, 0, :, :, :]

    def load_data_processed(self):
        return self.memmap_file[:, 1, :, :, :]

    # Returns the full (x2) memory for hardware to use
    def load_trial_all_data(self, trial):
        return self.memmap_file[trial, :, :, :, :]
