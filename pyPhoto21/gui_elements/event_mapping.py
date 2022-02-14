class EventMapping:

    def __init__(self, gui):
        self.event_mapping = {
            'Record': {
                'function': gui.record,
                'args': {}
            },
            'Take RLI': {
                'function': gui.take_rli,
                'args': {}
            },
            'Save': {
                'function': gui.data.save_metadata_to_json,
                'args': {}
            },
            "-CAMERA PROGRAM-": {
                'function': gui.set_camera_program,
                'args': {},
            },
            '-github-': {
                'function': gui.launch_github_page,
                'args': {'kind': "technical"},
            },
            'Help': {
                'function': gui.launch_github_page,
                'args': {'kind': "user"},
            },
            '-psg-': {
                'function': gui.launch_github_page,
                'args': {'kind': "issue"},
            },
            '-timer-': {
                'function': gui.launch_little_dave_calendar,
                'args': {'kind': "issue"},
            },
            "Choose Save Directory": {
                'function': gui.choose_save_dir,
                'args': {},
            },
            'Acquisition Onset': {
                'function': gui.set_acqui_onset,
                'args': {},
            },
            'Acquisition Duration': {
                'function': gui.set_acqui_duration,
                'args': {},
            },
            'Stimulator #1 Onset': {
                'function': gui.validate_and_pass_channel,
                'args': {'channel': 1, 'call': gui.data.hardware.set_stim_onset},
            },
            'Shutter Onset': {
                'function': gui.validate_and_pass_channel,
                'args': {'channel': None, 'call': gui.data.hardware.set_shutter_onset},
            },
            'Stimulator #2 Onset': {
                'function': gui.validate_and_pass_channel,
                'args': {'channel': 2, 'call': gui.data.hardware.set_stim_onset},
            },
            'Stimulator #1 Duration': {
                'function': gui.validate_and_pass_channel,
                'args': {'channel': 1, 'call': gui.data.hardware.set_stim_duration},
            },
            'Stimulator #2 Duration': {
                'function': gui.validate_and_pass_channel,
                'args': {'channel': 2, 'call': gui.data.hardware.set_stim_duration},
            },
            'num_pulses Stim #1': {
                'function': gui.validate_and_pass_channel,
                'args': {'channel': 1, 'call': gui.data.hardware.set_num_pulses},
            },
            'num_pulses Stim #2': {
                'function': gui.validate_and_pass_channel,
                'args': {'channel': 2, 'call': gui.data.hardware.set_num_pulses},
            },
            'int_pulses Stim #1': {
                'function': gui.validate_and_pass_channel,
                'args': {'channel': 1, 'call': gui.data.hardware.set_int_pulses},
            },
            'int_pulses Stim #2': {
                'function': gui.validate_and_pass_channel,
                'args': {'channel': 2, 'call': gui.data.hardware.set_int_pulses},
            },
            'num_bursts Stim #1': {
                'function': gui.validate_and_pass_channel,
                'args': {'channel': 1, 'call': gui.data.hardware.set_num_bursts},
            },
            'num_bursts Stim #2': {
                'function': gui.validate_and_pass_channel,
                'args': {'channel': 2, 'call': gui.data.hardware.set_num_bursts},
            },
            'int_bursts Stim #1': {
                'function': gui.validate_and_pass_channel,
                'args': {'channel': 1, 'call': gui.data.hardware.set_int_bursts},
            },
            'int_bursts Stim #2': {
                'function': gui.validate_and_pass_channel,
                'args': {'channel': 2, 'call': gui.data.hardware.set_int_bursts},
            },
            'num_trials': {
                'function': gui.validate_and_pass_int,
                'args': {'call': gui.data.set_num_trials, 'max_val': 20},
            },
            'int_trials': {
                'function': gui.validate_and_pass_int,
                'args': {'call': gui.data.set_int_trials},
            },
            'Number of Points': {
                'function': gui.set_num_pts,
                'args': {}
            },
            'num_records': {
                'function': gui.validate_and_pass_int,
                'args': {'call': gui.data.set_num_records, 'max_val': 20},
            },
            'int_records': {
                'function': gui.validate_and_pass_int,
                'args': {'call': gui.data.set_int_records},
            },
            "Increment Trial": {
                'function': gui.pass_no_arg_calls,
                'args': {'call': gui.data.increment_current_trial_index,
                         'call2': gui.update_tracking_num_fields}
            },
            "Decrement Trial": {
                'function': gui.pass_no_arg_calls,
                'args': {'call': gui.data.decrement_current_trial_index,
                         'call2': gui.update_tracking_num_fields}
            },
            "Increment Record": {
                'function': gui.pass_no_arg_calls,
                'args': {'call': gui.data.increment_record,
                         'call2': gui.update_tracking_num_fields}
            },
            "Decrement Record": {
                'function': gui.pass_no_arg_calls,
                'args': {'call': gui.data.decrement_record,
                         'call2': gui.update_tracking_num_fields}
            },
            "Increment Location": {
                'function': gui.pass_no_arg_calls,
                'args': {'call': gui.data.increment_location,
                         'call2': gui.update_tracking_num_fields}
            },
            "Decrement Location": {
                'function': gui.pass_no_arg_calls,
                'args': {'call': gui.data.decrement_location,
                         'call2': gui.update_tracking_num_fields}
            },
            "Increment Slice": {
                'function': gui.pass_no_arg_calls,
                'args': {'call': gui.data.increment_slice,
                         'call2': gui.update_tracking_num_fields}
            },
            "Decrement Slice": {
                'function': gui.pass_no_arg_calls,
                'args': {'call': gui.data.decrement_slice,
                         'call2': gui.update_tracking_num_fields}
            },
            "Increment File": {
                'function': gui.pass_no_arg_calls,
                'args': {'call': gui.data.increment_file,
                         'call3': gui.sync_gui_fields_from_meta}
            },
            "Decrement File": {
                'function': gui.pass_no_arg_calls,
                'args': {'call': gui.data.decrement_file,
                         'call3': gui.sync_gui_fields_from_meta}
            },
            "Trial Number": {
                'function': gui.validate_and_pass_int,
                'args': {'call': gui.set_current_trial_index,
                         'call2': gui.update_tracking_num_fields}
            },
            "Location Number": {
                'function': gui.validate_and_pass_int,
                'args': {'call': gui.set_location,
                         'call2': gui.update_tracking_num_fields}
            },
            "Record Number": {
                'function': gui.validate_and_pass_int,
                'args': {'call': gui.set_record,
                         'call2': gui.update_tracking_num_fields}
            },
            "Slice Number": {
                'function': gui.validate_and_pass_int,
                'args': {'call': gui.set_slice,
                         'call2': gui.update_tracking_num_fields}
            },
            "STOP!": {
                'function': gui.hardware.set_stop_flag,
                'args': {}
            },

            "Reset Cam": {
                'function': gui.hardware.reset_camera,
                'args': {}
            },
            "Save Preference": {
                'function': gui.save_preference,
                'args': {}
            },
            "Load Preference": {
                'function': gui.load_preference,
                'args': {}
            },
            "About": {
                'function': gui.introduction,
                'args': {}
            },
        }

    def get_event_mapping(self):
        return self.event_mapping
