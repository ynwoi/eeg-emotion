import os
from typing import Any, Callable, Dict, Tuple, Union

import scipy.io as scio

from dataloaders.base_dataset import BaseDataset


class SEEDDataset(BaseDataset):


    def __init__(self,
                 root_path: str = './Preprocessed_EEG',
                 chunk_size: int = 200,
                 overlap: int = 0,
                 num_channel: int = 62,
                 online_transform: Union[None, Callable] = None,
                 offline_transform: Union[None, Callable] = None,
                 label_transform: Union[None, Callable] = None,
                 before_trial: Union[None, Callable] = None,
                 after_trial: Union[Callable, None] = None,
                 after_session: Union[Callable, None] = None,
                 after_subject: Union[Callable, None] = None,
                 io_path: str = './io/seed',
                 io_size: int = 10485760,
                 io_mode: str = 'lmdb',
                 num_worker: int = 0,
                 verbose: bool = True,
                 in_memory: bool = False):
        # pass all arguments to super class
        params = {
            'root_path': root_path,
            'chunk_size': chunk_size,
            'overlap': overlap,
            'num_channel': num_channel,
            'online_transform': online_transform,
            'offline_transform': offline_transform,
            'label_transform': label_transform,
            'before_trial': before_trial,
            'after_trial': after_trial,
            'after_session': after_session,
            'after_subject': after_subject,
            'io_path': io_path,
            'io_size': io_size,
            'io_mode': io_mode,
            'num_worker': num_worker,
            'verbose': verbose,
            'in_memory': in_memory
        }
        super().__init__(**params)
        # save all arguments to __dict__
        self.__dict__.update(params)

    @staticmethod
    def process_record(file: Any = None,
                   root_path: str = './Preprocessed_EEG',
                   chunk_size: int = 200,
                   overlap: int = 0,
                   num_channel: int = 62,
                   before_trial: Union[None, Callable] = None,
                   offline_transform: Union[None, Callable] = None,
                   after_trial: Union[None, Callable] = None,
                   **kwargs):
        file_name = file

        subject = int(os.path.basename(file_name).split('.')[0].split('_')
                      [0])  # subject (15)
        date = int(os.path.basename(file_name).split('.')[0].split('_')
                   [1])  # period (3)

        samples = scio.loadmat(os.path.join(root_path, file_name),
                               verify_compressed_data_integrity=False
                               )  # trial (15), channel(62), timestep(n*200)
        # label file
        labels = scio.loadmat(
            os.path.join(root_path, 'label.mat'),
            verify_compressed_data_integrity=False)['label'][0]

        trial_ids = [key for key in samples.keys() if 'eeg' in key]
        
        write_pointer = 0
        # loop for each trial
        for trial_id in trial_ids:
            
            # extract baseline signals
            trial_samples = samples[trial_id]  # channel(62), timestep(n*200)
            if before_trial:
                trial_samples = before_trial(trial_samples)

            # record the common meta info
            trial_meta_info = {
                'subject_id': subject,
                'trial_id': trial_id,
                'emotion': int(labels[int(trial_id.split('_')[-1][3:]) - 1]),
                'date': date
            }

            # extract experimental signals
            start_at = 0
            if chunk_size <= 0:
                dynamic_chunk_size = trial_samples.shape[1] - start_at
            else:
                dynamic_chunk_size = chunk_size

            # chunk with chunk size
            end_at = dynamic_chunk_size
            # calculate moving step
            step = dynamic_chunk_size - overlap

            while end_at <= trial_samples.shape[1]:
                clip_sample = trial_samples[:num_channel, start_at:end_at]

                t_eeg = clip_sample
                if not offline_transform is None:
                    t_eeg = offline_transform(eeg=clip_sample)['eeg']

                clip_id = f'{file_name}_{write_pointer}'
                write_pointer += 1

                # record meta info for each signal
                record_info = {
                    'start_at': start_at,
                    'end_at': end_at,
                    'clip_id': clip_id
                }
                record_info.update(trial_meta_info)
                yield {'eeg': t_eeg, 'key': clip_id, 'info': record_info}

                start_at = start_at + step
                end_at = start_at + dynamic_chunk_size

    def set_records(self, root_path: str = './Preprocessed_EEG', **kwargs):
        file_list = os.listdir(root_path)
        skip_set = ['label.mat', 'readme.txt']
        file_list = [f for f in file_list if f not in skip_set]
        return file_list

    def __getitem__(self, index: int) -> Tuple[any, any, int, int, int]:
        info = self.read_info(index)

        eeg_index = str(info['clip_id'])
        eeg_record = str(info['_record_id'])
        eeg = self.read_eeg(eeg_record, eeg_index)

        signal = eeg
        label = info

        if self.online_transform:
            signal = self.online_transform(eeg=eeg)['eeg']

        if self.label_transform:
            label = self.label_transform(y=info)['y']

        return signal, label

    @property
    def repr_body(self) -> Dict:
        return dict(
            super().repr_body, **{
                'root_path': self.root_path,
                'chunk_size': self.chunk_size,
                'overlap': self.overlap,
                'num_channel': self.num_channel,
                'online_transform': self.online_transform,
                'offline_transform': self.offline_transform,
                'label_transform': self.label_transform,
                'before_trial': self.before_trial,
                'after_trial': self.after_trial,
                'num_worker': self.num_worker,
                'verbose': self.verbose,
                'io_size': self.io_size
            })
