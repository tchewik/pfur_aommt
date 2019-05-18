"""
    Implemented for Python 3.6+
"""

import numpy as np
import struct
from .fourier_transform import Fourier
import copy
from .alpha_mu_laws import AlphaLaw, MuLaw


class WAVAudio:
    def __init__(self, meta_info, *data):
        self.meta_info = meta_info
        self.data = data

    def __repr__(self):
        return str(self.meta_info)

    def separate_channels(self):
        if self.meta_info['num_channels'] == 2:
            return np.array([self.data[::2], self.data[1::2]])

    def channels_to_row(self, separate_channels):
        flatten = lambda l: [item for sublist in l for item in sublist]
        return flatten([[channel[i] for channel in separate_channels] for i in range(len(separate_channels[0]))])


class AudioProcessorWAVE:
    def __init__(self):
        self.fields = {
            'chunk_id': (4, False),
            'chunk_size': (4, True),
            'format': (4, False),
            'subchunk_1_id': (4, False),
            'subchunk_1_size': (4, True),
            'audio_format': (2, True),
            'num_channels': (2, True),
            'sample_rate': (4, True),
            'byte_rate': (4, True),
            'block_align': (2, True),
            'bits_per_sample': (2, True),
            'subchunk_2_id': (4, False),
            'subchunk_2_size': (4, True)}
        self.integers = ['chunk_size', 'subchunk_1_size', 'audio_format', 'num_channels', 'sample_rate', 'byte_rate',
                         'block_align', 'bits_per_sample']

    def read(self, filename):
        with open(filename, 'rb') as audio_file:
            lines = audio_file.read()

        offset = 0
        meta_info = {}
        for key in self.fields.keys():
            meta_info[key] = lines[offset:offset + self.fields[key][0]]
            if self.fields[key][1]:
                meta_info[key] = int.from_bytes(meta_info[key], 'little')
            offset += self.fields[key][0]

        data = lines[offset:offset + meta_info['subchunk_2_size']]
        new_audio = WAVAudio(meta_info, *data)

        return new_audio

    def write(self, audio: WAVAudio, filename):
        with open(filename, 'wb') as audio_file:
            for key in audio.meta_info:
                value = audio.meta_info[key]
                fmt = 'i' if self.fields[key][0] == 4 else 'h'

                if type(value) == int:
                    audio_file.write(struct.pack(fmt, value))
                elif type(value) == bytes:
                    audio_file.write(value)

            if type(audio.data[0]) == int:
                for byte in audio.data:
                    audio_file.write(struct.pack('B', byte))
            else:
                audio_file.write(struct.pack('f' * len(audio.data), *audio.data))

    def calc_dft(self, audio):
        dft = Fourier.dft(audio.data)
        return dft

    def calc_idft(self, audio):
        idft = Fourier.idft(audio.data)
        return np.real(idft).astype(int)

    def apply_hanna_window(self, audio):
        window = Fourier._hanna_window(audio.data)
        return window

    def mu_law(self, audio, encode=True):
        new_audio = copy.copy(audio)
        processor = MuLaw()
        if encode:
            new_audio.data = processor.encode(audio.data)
        else:
            new_audio.data = processor.decode(audio.data)
        return new_audio

    def alpha_law(self, audio, encode=True):
        new_audio = copy.copy(audio)
        processor = AlphaLaw()
        if encode:
            new_audio.data = processor.encode(audio.data)
        else:
            new_audio.data = processor.decode(audio.data)
        return new_audio
