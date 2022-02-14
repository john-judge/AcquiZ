import numpy as np
import threading
import time
import PySimpleGUI as sg
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from webbrowser import open as open_browser

from pyPhoto21.viewers.daq import DAQViewer
from pyPhoto21.gui_elements.layouts import *
from pyPhoto21.gui_elements.event_mapping import EventMapping


class GUI:

    def __init__(self, data, production_mode=True):
        matplotlib.use("TkAgg")
        sg.theme('DarkBlue12')
        self.data = data
        data.gui = self
        self.hardware = data.hardware
        self.production_mode = production_mode
        self.dv = DAQViewer(self.data)
        self.layouts = Layouts(data)
        self.window = None

        # general state/settings
        self.title = "Photo21"
        self.freeze_input = False  # whether to allow fields to be updated. Frozen during acquire (how about during file loaded?)
        self.event_mapping = None
        self.define_event_mapping()  # event callbacks used in event loops
        self.cached_num_pts = 600  # number of points to restore to whenever not 200 Hz cam program
        self.cached_num_trials = 5
        # kickoff workflow
        if self.production_mode:
            self.introduction()
        self.main_workflow()

    def introduction(self):
        layout = [[
            sg.Column([[sg.Image(key="-IMAGE-")]]),
            [sg.Text("Welcome to Photo21! \n\tCheck that your camera and \n\tNI-USB are turned on.")],
            [sg.Button("OK")]
        ]]
        intro_window = sg.Window(self.title, layout, finalize=True)
        self.intro_event_loop(intro_window)
        intro_window.close()

    @staticmethod
    def intro_event_loop(window, filename='art/meyer.png'):
        window["-IMAGE-"].update(filename=filename)
        while True:
            event, values = window.read()
            # End intro when user closes window or
            # presses the OK button
            if event == "OK" or event == sg.WIN_CLOSED:
                break

    def main_workflow(self):
        right_col = self.layouts.create_right_column(self)
        left_col = self.layouts.create_left_column(self)
        toolbar_menu = self.layouts.create_menu()

        layout = [[toolbar_menu],
                  [sg.Column(left_col),
                   sg.VSeperator(),
                   sg.Column(right_col)]]

        self.window = sg.Window(self.title,
                                layout,
                                finalize=True,
                                element_justification='center',
                                resizable=True,
                                font='Helvetica 18')
        self.plot_daq_timeline()
        self.main_workflow_loop()
        self.window.close()

    def main_workflow_loop(self, history_debug=False, window=None, exit_event="Exit"):
        if window is None:
            window = self.window
        events = ''
        while True:
            event, values = window.read()
            if history_debug and event is not None and not self.production_mode:
                events += str(event) + '\n'
            if event == exit_event or event == sg.WIN_CLOSED or event == '-close-':
                if self.is_recording():
                    self.data.save_metadata_to_json()
                    print("Cleaning up hardware before exiting. Waiting until safe to exit (or at most 3 seconds)...")

                    self.hardware.set_stop_flag(True)
                    timeout = 3
                    while self.hardware.get_stop_flag() and timeout > 0:
                        time.sleep(1)
                        timeout -= 1
                        print(timeout, "seconds")
                break
            elif event not in self.event_mapping or self.event_mapping[event] is None:
                print("Not Implemented:", event)
            else:
                ev = self.event_mapping[event]
                if event in values:
                    ev['args']['window'] = window
                    ev['args']['values'] = values[event]
                    ev['args']['event'] = event
                ev['function'](**ev['args'])
        if history_debug and not self.production_mode:
            print("**** History of Events ****\n", events)

    def is_recording(self):
        return self.freeze_input and not self.data.get_is_loaded_from_file()

    @staticmethod
    def draw_figure(canvas, fig):
        if canvas.children:
            for child in canvas.winfo_children():
                child.destroy()
        figure_canvas_agg = FigureCanvasTkAgg(fig, master=canvas)
        figure_canvas_agg.draw_idle()
        figure_canvas_agg.get_tk_widget().pack(fill='none', expand=True)
        return figure_canvas_agg

    def get_trial_sleep_time(self):
        sleep_sec = self.data.get_int_trials()
        if self.data.get_is_schedule_rli_enabled():
            sleep_sec = max(0, sleep_sec - .12)  # attempt to shorten by 120 ms, rough lower bound on time to take RLI
        return sleep_sec

    def get_record_sleep_time(self):
        sleep_sec = self.data.get_int_records()
        return max(0, sleep_sec - self.get_trial_sleep_time())

    # returns True if stop flag is set
    def sleep_and_check_stop_flag(self, sleep_time, interval=1):
        elapsed = 0
        while elapsed < sleep_time:
            time.sleep(interval)
            elapsed += interval
            if self.hardware.get_stop_flag():
                self.hardware.set_stop_flag(False)
                return True
        return False

    def record_in_background(self):
        self.hardware.set_stop_flag(False)

        sleep_trial = self.get_trial_sleep_time()
        sleep_record = self.get_record_sleep_time()

        self.data.set_is_loaded_from_file(False)
        exit_recording = False

        if self.data.get_num_records() * self.data.get_num_trials() * self.data.get_num_pts() == 0:
            print("Settings are such that no trials or points are recorded. Ending recording session.")
            exit_recording = True
        if self.data.get_num_pts() <= 10:
            self.notify_window("Too few points",
                               "Please increase number of points to record. NI-DAQmx may fail to"
                               " sample or administer stimulation with"
                               " too few points.")
            exit_recording = True

        # Note that record index may not necessarily match the record num for file saving
        for record_index in range(self.data.get_num_records()):
            is_last_record = (record_index == self.data.get_num_records() - 1)
            if exit_recording:
                break

            for trial in range(self.data.get_num_trials()):
                self.data.set_current_trial_index(trial)

                is_last_trial = (trial == self.data.get_num_trials() - 1)
                if self.data.get_is_schedule_rli_enabled():
                    self.take_rli_core()
                acqui_mem = self.data.get_acqui_memory()
                self.hardware.record(images=acqui_mem,
                                     fp_data=self.data.get_fp_data())

                self.update_tracking_num_fields()
                print("\tTook trial", trial + 1, "of", self.data.get_num_trials())
                if not is_last_trial:
                    print("\t\t", sleep_trial, "seconds until next trial...")

                    exit_recording = self.sleep_and_check_stop_flag(sleep_time=sleep_trial)
                if exit_recording:
                    break

            self.data.drop_processing_lock()
            time.sleep(0.2)

            self.data.save_metadata_to_json()
            self.update_tracking_num_fields()
            if exit_recording:
                break
            print("Took recording set", record_index + 1, "of", self.data.get_num_records())
            if not is_last_record:
                self.data.increment_record_until_filename_free()
                print("\t", sleep_record, "seconds until next recording set...")
                exit_recording = self.sleep_and_check_stop_flag(sleep_time=sleep_record)

        print("Recording ended.")
        # done recording

    def record(self, **kwargs):
        # we spawn a new thread to acquire in the background.
        # meanwhile the original thread returns and keeps handling GUI clicks
        # but updates to Data/Hardware fields will be frozen
        # self.record_in_background()
        if not self.data.get_is_schedule_rli_enabled():
            self.notify_window("Auto RLI not enabled",
                               "RLI will not be automatically collected for each recording set.")
        if self.data.is_save_dir_default():
            self.notify_window("Save Folder",
                               "You haven't chosen a folder to contain new files."
                               " \nLet's choose a save folder for this recording session.")
            self.choose_save_dir()
        self.data.save_metadata_to_json()
        threading.Thread(target=self.record_in_background, args=(), daemon=True).start()

    ''' RLI Controller functions '''

    def take_rli_core(self):
        self.hardware.take_rli(images=self.data.get_rli_memory())
        self.data.calculate_rli(force_recalculate=True)
        self.data.set_is_loaded_from_file(False)
        self.data.save_metadata_to_json()

    def take_rli_in_background(self):
        self.hardware.set_stop_flag(False)
        self.take_rli_core()

    def take_rli(self, **kwargs):
        self.data.get_current_trial_index()
        threading.Thread(target=self.take_rli_in_background, args=(), daemon=True).start()

    def set_camera_program(self, **kwargs):
        curr_program = self.data.get_camera_program()
        program_name = kwargs['values']
        program_index = self.data.display_camera_programs.index(program_name)
        if program_index == 0 and curr_program != 0:
            self.cached_num_pts = self.data.get_num_pts()
            self.cached_num_trials = self.data.get_num_trials()
            self.data.set_num_trials(1)
            self.window["num_trials"].update('1')
            self.set_num_pts(values='50', suppress_resize=True)
            self.data.set_num_dark_rli(1, prevent_resize=True)
            self.data.set_num_light_rli(1, prevent_resize=True)
        elif curr_program == 0 and program_index != 0:
            print("getting cached settings...")
            self.set_num_pts(values=str(self.cached_num_pts), suppress_resize=True)
            self.data.set_num_trials(self.cached_num_trials)
            self.window["num_trials"].update(str(self.cached_num_trials))
            self.data.set_num_dark_rli(200, prevent_resize=True)
            self.data.set_num_light_rli(280, prevent_resize=True)
        self.data.set_camera_program(program_index)
        self.update_tracking_num_fields()
        self.window["Acquisition Duration"].update(self.data.get_acqui_duration())

    @staticmethod
    def notify_window(title, message):
        layout = [[sg.Column([
            [sg.Text(message)],
            [sg.Button("OK")]])]]
        wind = sg.Window(title, layout, finalize=True)
        while True:
            event, values = wind.read()
            # End intro when user closes window or
            # presses the OK button
            if event == "OK" or event == sg.WIN_CLOSED:
                break
        wind.close()

    def choose_save_dir(self, **kwargs):
        folder = self.browse_for_folder()
        if folder is not None:
            self.data.set_save_dir(folder)
            self.data.db.set_save_dir(folder)
            print("New save location:", folder)

    def browse_for_file(self, file_extensions, multi_file=False, tsv_only=False):
        layout_choice = None
        if not multi_file:
            layout_choice = self.layouts.create_file_browser()
        else:
            layout_choice = self.layouts.create_files_browser()
        file_window = sg.Window('File Browser',
                                layout_choice,
                                finalize=True,
                                element_justification='center',
                                resizable=True,
                                font='Helvetica 18')
        file = None
        # file browser event loop
        while True:
            event, values = file_window.read()
            if event == sg.WIN_CLOSED or event == "Exit":
                file_window.close()
                return
            elif event == "file_window.open":
                file = values["file_window.browse"]
                file_ext = file.split('.')
                if len(file_ext) > 0:
                    file_ext = file_ext[-1]
                else:
                    file_ext = ''
                if file_ext not in file_extensions:
                    supported_file_str = " ".join(file_extensions)
                    self.notify_window("File Type",
                                       "Unsupported file type.\nSupported: " + supported_file_str)
                else:
                    break
        if self.freeze_input and not self.data.get_is_loaded_from_file():
            file = None
            self.notify_window("File Input Error",
                               "Cannot load file during acquisition")
        file_window.close()
        return file

    def browse_for_save_as_file(self, file_types=(("Tab-Separated Value file", "*.tsv"),)):
        w = sg.Window('Save As',
                      self.layouts.create_save_as_browser(file_types),
                      finalize=True,
                      element_justification='center',
                      resizable=True,
                      font='Helvetica 18')
        new_file = None
        # file browser event loop
        while True:
            event, values = w.read()
            if event == sg.WIN_CLOSED or event == "Exit":
                w.close()
                return
            elif event == "save_as_window.open":
                new_file = values["save_as_window.browse"]
                break
        if self.is_recording():
            new_file = self.data.get_save_dir()
            self.notify_window("Warning",
                               "Please stop recording before exporting data.")
            new_file = None
        w.close()
        if new_file is None or len(new_file) < 1:
            return None
        return new_file

    def browse_for_folder(self, recording_notif=True):
        folder_window = sg.Window('Folder Browser',
                                  self.layouts.create_folder_browser(),
                                  finalize=True,
                                  element_justification='center',
                                  resizable=True,
                                  font='Helvetica 18')
        folder = None
        # file browser event loop
        while True:
            event, values = folder_window.read()
            if event == sg.WIN_CLOSED or event == "Exit":
                folder_window.close()
                return
            elif event == "folder_window.open":
                folder = values["folder_window.browse"]
                break
        if recording_notif and self.is_recording():
            folder = self.data.get_save_dir()
            self.notify_window("Warning",
                               "You are changing the save location during acquisition." +
                               "I don't recommend scattering your files. " +
                               "Keeping this save directory:\n" +
                               folder)
        folder_window.close()
        if len(folder) < 1:
            return None
        return folder

    def load_preference(self):
        file = self.browse_for_file(['json'])
        if file is not None:
            print("Loading from preference file:", file)
            self.data.load_preference_file(file)
            self.sync_gui_fields_from_meta()
            self.dv.update()

    def save_preference(self):
        file = self.browse_for_save_as_file(('JSON', "*" + self.data.metadata_extension))
        if file is not None:
            self.data.save_preference_file(file)

    # Pull all file-based data from Data and sync GUI fields
    def sync_gui_fields_from_meta(self):
        w = self.window

        # Hardware settings
        w['Number of Points'].update(self.data.get_num_pts())
        w['int_records'].update(self.data.get_int_records())
        w['num_records'].update(self.data.get_num_records())
        w['Acquisition Onset'].update(self.data.get_acqui_onset())
        w['Acquisition Duration'].update(self.data.get_acqui_duration())
        w['Stimulator #1 Onset'].update(self.data.get_stim_onset(1))
        w['Stimulator #2 Onset'].update(self.data.get_stim_onset(2))
        w['Stimulator #1 Duration'].update(self.data.get_stim_duration(1))
        w['Stimulator #2 Duration'].update(self.data.get_stim_duration(2))
        w['int_trials'].update(self.data.get_int_trials())
        w['num_trials'].update(self.data.get_num_trials())
        w['-CAMERA PROGRAM-'].update(self.data.display_camera_programs[self.data.get_camera_program()])
        self.dv.update()


    @staticmethod
    def launch_github_page(**kwargs):
        urls = {
            'technical': 'https://github.com/john-judge/PhotoLib#photolib',
            'user': 'https://github.com/john-judge/PhotoLib/blob/master/'
                    'TUTORIAL.md#users-manual-for-pyphoto21-little-dave',  # Update this to user tutorial link
            'issue': 'https://github.com/john-judge/PhotoLib/issues/new'
        }
        if 'kind' in kwargs and kwargs['kind'] in urls:
            open_browser(urls[kwargs['kind']], new=2)

    @staticmethod
    def launch_youtube_tutorial():
        pass

    @staticmethod
    def launch_little_dave_calendar(**kwargs):
        open_browser('https://calendar.google.com/calendar'
                     '/render?cid=24tfud764rqbe4tcdgvqmi6pdc@'
                     'group.calendar.google.com')

    # Returns True if string s is a valid numeric input
    @staticmethod
    def validate_numeric_input(s, non_zero=False, max_digits=None, min_val=None, max_val=None, decimal=False):
        if decimal:  # decimals: allow removing at most one decimal anywhere
            if len(s) > 0 and s[0] == '.':
                s = s[1:]
            elif len(s) > 0 and s[-1] == '.':
                s = s[:-1]
            elif '.' in s:
                s = s.replace('.', '')
        return type(s) == str \
               and s.isnumeric() \
               and (max_digits is None or len(s) <= max_digits) \
               and (not non_zero or int(s) != 0) \
               and (min_val is None or int(s) >= min_val) \
               and (max_val is None or int(s) <= max_val)

    def set_acqui_onset(self, **kwargs):
        v = kwargs['values']
        while len(v) > 0 and not self.validate_numeric_input(v, decimal=True, max_val=5000):
            v = v[:-1]
        if self.validate_numeric_input(v, decimal=True, max_val=5000):
            num_frames = float(v) // self.data.get_int_pts()
            self.data.set_acqui_onset(float(num_frames))
            self.window['Acquisition Onset'].update(v)
            self.dv.update()

    def set_num_pts(self, suppress_resize=False, **kwargs):
        v = kwargs['values']

        while len(v) > 0 and not self.validate_numeric_input(v, decimal=True, max_val=5000):
            v = v[:-1]
        if len(v) > 0 and self.validate_numeric_input(v, decimal=True, max_val=5000):
            acqui_duration = float(v) * self.data.get_int_pts()
            self.data.set_num_pts(value=int(v), prevent_resize=suppress_resize)  # Data method resizes data
            self.window["Number of Points"].update(v)
            self.window["Acquisition Duration"].update(str(acqui_duration))
        else:
            self.data.set_num_pts(value=0, prevent_resize=suppress_resize)  # Data method resizes data
            self.window["Number of Points"].update('')
            self.window["Acquisition Duration"].update('')
        self.dv.update()
        self.update_tracking_num_fields(no_plot_update=True)
        if self.data.core.get_is_temporal_filter_enabled():
            filter_type = self.data.core.get_temporal_filter_options()[
                self.data.core.get_temporal_filter_index()]
            sigma_t = self.data.core.get_temporal_filter_radius()
            if not self.data.validate_filter_size(filter_type, sigma_t):
                self.notify_window("Invalid Settings",
                                   "Measure window is too small for the"
                                   " default cropping needed for the temporal filter"
                                   " settings. \nUntil measure window is widened or "
                                   " filtering radius is decreased, temporal filtering will"
                                   " not be applied to traces.")

    def set_acqui_duration(self, **kwargs):
        v = kwargs['values']
        min_pts = 10

        # looks at num_pts as well to validate.
        def is_valid_acqui_duration(u, max_num_pts=15000):
            return self.validate_numeric_input(u, decimal=True) \
                   and int(float(u) * self.data.get_int_pts()) <= max_num_pts

        while len(v) > 0 and not is_valid_acqui_duration(v):
            v = v[:-1]
        if len(v) > 0 and is_valid_acqui_duration(v):
            num_pts = int(float(v) // self.data.get_int_pts())
            self.data.set_num_pts(value=num_pts)
            self.window["Acquisition Duration"].update(v)
            self.window["Number of Points"].update(str(num_pts))
        else:
            self.data.set_num_pts(value=0)
            self.window["Acquisition Duration"].update('')
            self.window["Number of Points"].update('')
        self.dv.update()

    def plot_daq_timeline(self):
        fig = self.dv.get_fig()
        self.draw_figure(self.window['daq_canvas'].TKCanvas, fig)
        self.dv.update()

    @staticmethod
    def pass_no_arg_calls(**kwargs):
        for key in kwargs:
            if key.startswith('call'):
                kwargs[key]()

    def validate_and_pass_int(self, **kwargs):
        max_val = None
        if 'max_val' in kwargs:
            max_val = kwargs['max_val']
        fn_to_call = kwargs['call']
        v = kwargs['values']
        window = kwargs['window']
        while len(v) > 0 and not self.validate_numeric_input(v, max_digits=5, max_val=max_val):
            v = v[:-1]
        if len(v) > 0 and self.validate_numeric_input(v, max_digits=5, max_val=max_val):
            fn_to_call(value=int(v))
            window[kwargs['event']].update(v)
            if not self.production_mode:
                print("called:", fn_to_call)
            if 'call2' in kwargs:
                kwargs['call2'](value=int(v))
                if not self.production_mode:
                    print("called:", kwargs['call2'])
        else:
            fn_to_call(value=None)
            window[kwargs['event']].update('')

    # for passing to channel-based setters
    def validate_and_pass_channel(self, **kwargs):
        fns_to_call = []
        for k in kwargs:
            if k.startswith('call'):
                fns_to_call.append(kwargs[k])
        v = kwargs['values']
        ch = kwargs['channel']
        window = kwargs['window']
        while len(v) > 0 and not self.validate_numeric_input(v, max_digits=6):
            v = v[:-1]
        if len(v) > 0 and self.validate_numeric_input(v, max_digits=6):
            for fn in fns_to_call:
                fn(value=int(v), channel=ch)
            window[kwargs['event']].update(v)
            if not self.production_mode:
                print("called:", fns_to_call)
        else:
            for fn in fns_to_call:
                fn(value=0, channel=ch)
            window[kwargs['event']].update('')

        # update DAQ timeline visualization
        self.dv.update()

    def set_num_trials(self, **kwargs):
        v = kwargs['values']
        self.data.set_num_trials(int(v))

    def define_event_mapping(self):
        if self.event_mapping is None:
            self.event_mapping = EventMapping(self).get_event_mapping()

    def update_tracking_num_fields(self, no_plot_update=False, **kwargs):
        self.window["Slice Number"].update(self.data.get_slice_num())
        self.window["Location Number"].update(self.data.get_location_num())
        self.window["Record Number"].update(self.data.get_record_num())
        self.window["Trial Number"].update(self.data.get_current_trial_index())
        self.window["File Name"].update(self.data.db.get_current_filename(no_path=True,
                                                                          extension=self.data.db.extension))

    def set_current_trial_index(self, **kwargs):
        if 'value' in kwargs:
            if kwargs['value'] is None:
                value = None
            else:
                value = int(kwargs['value'])
            self.data.set_current_trial_index(value)

    def set_slice(self, **kwargs):
        value = int(kwargs['value'])
        self.data.set_slice(value)

    def set_record(self, **kwargs):
        value = int(kwargs['value'])
        self.data.set_record(value)

    def set_location(self, **kwargs):
        value = int(kwargs['value'])
        self.data.set_location(value)
