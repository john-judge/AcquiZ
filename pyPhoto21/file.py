import pandas as pd
import os, glob
import struct
import numpy as np


class File:

    def __init__(self, data):
        self.data = data
        self.current_slice = 0
        self.current_location = 0
        self.current_run = 0

    def increment_slice(self):
        self.current_slice += 1
        self.current_location = 0
        self.current_run = 0

    def increment_location(self):
        self.current_location += 1
        self.current_run = 0

    def increment_run(self):
        self.current_run += 1

    @staticmethod
    def pad_zero(i, dst_len=2):
        s = str(i)
        if len(s) < dst_len:
            return '0' + s
        return s

    def get_filename(self, extension='.zda'):
        return self.pad_zero(self.current_slice) + '-' + \
               self.pad_zero(self.current_location) + '-' + \
               self.pad_zero(self.current_run) + extension

    def save_to_file(self, acqui_images, rli_images):
        fn = self.get_filename()
        print("Saving to file " + fn + "...")
        version = 6  # Little Dave version
        # TO DO: use ZDA format.

    def load_from_file(self, filename):
        ds = Dataset(filename)
        # Set meta data
        meta = ds.get_meta()

        # Recording load
        images = ds.get_data()
        self.data.set_acqui_images(images, from_file=True)

        # RLI load
        rli_dict = ds.get_rli()
        rli_shape = rli_dict['rli_low'].shape
        rli_images = np.zeros((3, rli_shape[0], rli_shape[1]))
        rli_images[0, :, :] = rli_dict['rli_low']
        rli_images[1, :, :] = rli_dict['rli_high']
        rli_images[2, :, :] = rli_dict['rli_max']
        self.data.set_rli_images(rli_images, from_file=True)

        # FP data load
        fp_data = ds.get_fp_data()
        self.data.set_fp_data(fp_data)
        self.data.set_num_fp(meta['num_fp'])

        if meta is not None:
            self.current_slice = ds.slice_number
            self.current_location = ds.location_number
            self.current_run = ds.record_number
            self.data.set_num_trials(ds.number_of_trials)
            new_num_pts = ds.points_per_trace
            if new_num_pts < 1 or new_num_pts is None:
                new_num_pts = images.shape[1]
            print("Number of pts: ", new_num_pts)
            self.data.set_num_pts(new_num_pts)
            self.data.set_num_trials(ds.number_of_trials)
            # There are only 3 (averaged) RLI frames in ZDA files:
            self.data.set_num_dark_rli(1)
            self.data.set_num_light_rli(2)

        return ds.get_meta()


class Dataset:

    def __init__(self, filename, x_range=[0, -1], y_range=[0, -1], t_range=[0, -1]):
        self.x_range = x_range
        self.y_range = y_range
        self.t_range = t_range
        self.data, metadata, self.rli, self.fp_data = self.read_zda_to_df(filename)
        self.filename = filename
        self.meta = metadata
        if self.meta is not None:
            self.version = metadata['version']
            self.slice_number = metadata['slice_number']
            self.location_number = metadata['location_number']
            self.record_number = metadata['record_number']
            self.camera_program = metadata['camera_program']
            self.number_of_trials = metadata['number_of_trials']
            self.interval_between_trials = metadata['interval_between_trials']
            self.acquisition_gain = metadata['acquisition_gain']
            self.points_per_trace = metadata['points_per_trace']
            self.time_RecControl = metadata['time_RecControl']
            self.reset_onset = metadata['reset_onset']
            self.reset_duration = metadata['reset_duration']
            self.shutter_onset = metadata['shutter_onset']
            self.shutter_duration = metadata['shutter_duration']
            self.stimulation1_onset = metadata['stimulation1_onset']
            self.stimulation1_duration = metadata['stimulation1_duration']
            self.stimulation2_onset = metadata['stimulation2_onset']
            self.stimulation2_duration = metadata['stimulation2_duration']
            self.acquisition_onset = metadata['acquisition_onset']
            self.interval_between_samples = metadata['interval_between_samples']
            self.raw_width = metadata['raw_width']
            self.raw_height = metadata['raw_height']

            # original dimensions
            self.original = {'raw_width': self.raw_width,
                             'raw_height': self.raw_height,
                             'points_per_trace': self.points_per_trace}

    def clip_data(self, x_range=[0, -1], y_range=[0, -1], t_range=[0, -1]):
        """ Imposes a range restriction on frames and/or traces """
        self.x_range = x_range
        self.y_range = y_range
        self.t_range = t_range

    def get_fp_data(self, trial=None):
        if trial is None:
            return self.fp_data
        return self.fp_data[trial, :, :]

    def get_unclipped_data(self, trial=None):
        """ Returns unclipped data """

        # Reset to unclipped values
        self.meta['points_per_trace'] = self.original['points_per_trace']
        self.meta['raw_width'] = self.original['raw_width']
        self.meta['raw_height'] = self.original['raw_height']

        self.raw_width = self.meta['raw_width']
        self.raw_height = self.meta['raw_height']
        self.points_per_trace = self.meta['points_per_trace']

        if trial is not None:
            return self.data[trial, :, :, :]
        return self.data

    def get_data(self, trial=None):
        """ Returns clipped data """

        # Set to clipped values
        self.meta['points_per_trace'] = self.t_range[1] - self.t_range[0]
        self.meta['raw_width'] = self.x_range[1] - self.x_range[0]
        self.meta['raw_height'] = self.y_range[1] - self.y_range[0]

        self.raw_width = self.meta['raw_width']
        self.raw_height = self.meta['raw_height']
        self.points_per_trace = self.meta['points_per_trace']

        if trial is not None:
            return self.data[trial,
                   self.x_range[0]:self.x_range[1],
                   self.y_range[0]:self.y_range[1],
                   self.t_range[0]:self.t_range[1]]
        else:
            return self.data[:,
                   self.x_range[0]:self.x_range[1],
                   self.y_range[0]:self.y_range[1],
                   self.t_range[0]:self.t_range[1]]

    def get_meta(self):
        """ Returns metadata dictionary. Mostly for legacy behavior. """
        return self.meta

    def get_rli(self):
        """ Returns RLI data """
        return self.rli

    def get_trial_data(self, trial_no):
        """ Returns array slice for trial number """
        raise NotImplementedError
        return

    def read_zda_to_df(self, zda_file):
        """ Reads ZDA file to dataframe, and returns
        metadata as a dict.
        ZDA files are a custom PhotoZ binary format that must be interpreted byte-
        by-byte """
        file = open(zda_file, 'rb')
        # data type sizes in bytes
        chSize = 1
        shSize = 2
        nSize = 4
        tSize = 8
        fSize = 4

        metadata = {}
        metadata['version'] = (file.read(chSize))
        metadata['slice_number'] = (file.read(shSize))
        metadata['location_number'] = (file.read(shSize))
        metadata['record_number'] = (file.read(shSize))
        metadata['camera_program'] = (file.read(nSize))

        metadata['number_of_trials'] = (file.read(chSize))
        metadata['interval_between_trials'] = (file.read(chSize))
        metadata['acquisition_gain'] = (file.read(shSize))
        metadata['points_per_trace'] = (file.read(nSize))
        metadata['time_RecControl'] = (file.read(tSize))

        metadata['reset_onset'] = struct.unpack('f', (file.read(fSize)))[0]
        metadata['reset_duration'] = struct.unpack('f', (file.read(fSize)))[0]
        metadata['shutter_onset'] = struct.unpack('f', (file.read(fSize)))[0]
        metadata['shutter_duration'] = struct.unpack('f', (file.read(fSize)))[0]

        metadata['stimulation1_onset'] = struct.unpack('f', (file.read(fSize)))[0]
        metadata['stimulation1_duration'] = struct.unpack('f', (file.read(fSize)))[0]
        metadata['stimulation2_onset'] = struct.unpack('f', (file.read(fSize)))[0]
        metadata['stimulation2_duration'] = struct.unpack('f', (file.read(fSize)))[0]

        metadata['acquisition_onset'] = struct.unpack('f', (file.read(fSize)))[0]
        metadata['interval_between_samples'] = struct.unpack('f', (file.read(fSize)))[0]
        metadata['raw_width'] = (file.read(nSize))
        metadata['raw_height'] = (file.read(nSize))

        # Bytes to Python data type
        for k in metadata:
            if k in ['interval_between_samples', ] or 'onset' in k or 'duration' in k:
                pass  # any additional float processing can go here
            elif k == 'time_RecControl':
                pass  # TO DO: timestamp processing
            else:
                metadata[k] = int.from_bytes(metadata[k], "little")  # endianness

        metadata['num_fp'] = 4  # Little Dave
        if metadata['version'] <= 5:
            metadata['num_fp'] = 8  # Little Joe and legacy versions

        print("ZDA Version:", metadata['version'], "=> inferring", metadata['num_fp'], "FPs.")

        file.seek(1024, 0)
        # RLI
        rli_shape = (metadata['raw_height'], metadata['raw_width'])
        rli = {}
        for key in ['rli_low', 'rli_high', 'rli_max']:
            rli[key] = np.zeros(rli_shape)
            if key != 'rli_low':  # rli_low appears to not have FP data?
                for k in range(metadata['num_fp']):
                    # discard FP data associated with RLI
                    _ = int.from_bytes(file.read(shSize), "little")
            for jh in range(metadata['raw_height']):
                for jw in range(metadata['raw_width']):
                    rli[key][jh, jw] = int.from_bytes(file.read(shSize), "little")

        raw_data = np.zeros((metadata['number_of_trials'],
                             metadata['points_per_trace'],
                             metadata['raw_height'],
                             metadata['raw_width'])).astype(int)
        fp_data = np.zeros((metadata['number_of_trials'],
                            metadata['points_per_trace'],
                            metadata['num_fp'])).astype(int)
        num_read = 0
        for i in range(metadata['number_of_trials']):
            for x in range(metadata['num_fp']):
                for k in range(metadata['points_per_trace']):
                    if k != 0 and i != 0:  # skip the first num_fp
                        pt = self.read_data(file, shSize, num_read)
                        if pt is None:
                            return raw_data, metadata, rli, fp_data
                        fp_data[i, k, x] = int.from_bytes(pt, "little")
                        num_read += 1
            for jh in range(metadata['raw_height']):
                for jw in range(metadata['raw_width']):
                    for k in range(metadata['points_per_trace']):
                        pt = self.read_data(file, shSize, num_read)
                        if pt is None:
                            return raw_data, metadata, rli, fp_data
                        raw_data[i, k, jh, jw] = int.from_bytes(pt, "little")
                        num_read += 1

        file.close()
        return raw_data, metadata, rli, fp_data

    @staticmethod
    def read_data(file, sz, num_read):
        pt = file.read(sz)
        if not pt:
            print("Ran out of points. Read:", num_read)
            file.close()
            return None
        return pt

class DataLoader:

    def __init__(self):
        self.all_data = {}  # maps file names to Dataset objects

    def select_data_by_keyword(self, keyword):
        """ Returns the data for the first file matching the keyword """
        for file in self.all_data:
            if keyword in file:
                return self.all_data[file]

    def load_all_zda(self, data_dir='.'):
        """ Loads all ZDA data in data_dir
            into a dictionary of dataframes and metadata """
        n_files_loaded = 0
        for dirName, subdirList, fileList in os.walk(data_dir, topdown=True):
            for file in fileList:
                file = str(dirName + "/" + file)
                if '.zda' == file[-4:]:
                    self.all_data[file] = Dataset(file)
                    n_files_loaded += 1

        print('Number of files loaded:', n_files_loaded)
        return self.all_data

    def get_dataset(self, filename):
        return self.all_data[filename]
