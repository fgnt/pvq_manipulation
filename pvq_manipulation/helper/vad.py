import numpy as np
import paderbox as pb
import padertorch as pt
import typing

from dataclasses import dataclass


@pb.utils.functional.partial_decorator
def conv_smoothing(signal, window_length=7, threshold=3):
    """

    Boundary effects are visible at beginning and end of signal.

    Examples:
        >>> conv_smoothing(np.array([False, True, True, True, False, False, False, True]), 3, 2)
        array([False,  True,  True,  True, False, False, False, False])

    Args:
        signal:
        window_length:
        threshold:

    Returns:

    """
    left_context = right_context = (window_length - 1) // 2
    if window_length % 2 == 0:
        right_context += 1
    act_conv = np.sum(pb.array.segment_axis(
        np.pad(signal, (left_context, right_context), mode='constant'),
        length=window_length, shift=1, axis=0, end='cut'
    ), axis=-1)
    # act_conv = np.convolve(signal, np.ones(window_length), 's')
    act = act_conv >= threshold
    assert act.shape == signal.shape, (act.shape, signal.shape)
    return act


@dataclass
class VAD(pt.Configurable):
    smoothing: typing.Optional[typing.Callable] = None

    def reset(self):
        """Override for a stateful VAD"""
        pass

    def compute_vad(self, signal, time_resolution=True):
        raise NotImplementedError()

    def vad_to_time(self, vad, time_length):
        raise NotImplementedError()

    def __call__(self, signal, time_resolution=True, reset=True):
        if reset:
            self.reset()

        vad = self.compute_vad(signal)

        if self.smoothing is not None:
            vad = pb.array.interval.ArrayInterval(self.smoothing(vad))

        if time_resolution:
            vad = self.vad_to_time(vad, time_length=signal.shape[-1])

        return vad


class EnergyVAD(VAD):
    def __init__(self, sample_rate, threshold=0.3):
        self.sample_rate = sample_rate
        self.threshold = threshold

    @staticmethod
    def remove_silence(signal, vad_mask):
        return signal[vad_mask == 1]

    def __call__(self, example):
        signal = example['audio_data']  # B T
        vad_mask = self.get_vad_mask(signal)
        signal = self.remove_silence(signal, vad_mask)
        example['audio_data'] = signal
        example['vad_mask'] = vad_mask
        return example

    def get_vad_mask(self, signal):
        window_size = int(0.1 * self.sample_rate + 1)

        half_context = (window_size - 1) // 2
        std = np.std(signal, axis=-1, keepdims=True)
        signal = signal - np.mean(signal, axis=-1, keepdims=True)
        signal = np.abs(signal)
        zeros = np.zeros(
            [
                signal.shape[0],
                half_context,
            ]
        )
        signal = np.concatenate([zeros, signal, zeros], axis=1)
        sliding_max = np.max(pb.array.segment_axis(
            signal,
            length=window_size, shift=1, axis=1, end='cut'
        ), axis=-1)
        return sliding_max > self.threshold * std


@dataclass
class ThresholdVAD(VAD):
    """
    Energy-based VAD for almost clean files. Tested on WSJ clean data by Lukas
    Drude.

    Attributes:
        threshold: Fraction of total signal standard deviation. Use 0.3 for
            (almost) clean files (SNR >= 20dB, think LibriTTS) and 0.7 for less
            clean files (think LibriSpeech).
        window_size: Size of sliding max window.
        sample_rate: Sampling rate of audio data.
        smoothing: Optional callable that uses a sliding window over the raw
            decision to return a smoothed VAD.
    """
    threshold: float = 0.3
    window_size: typing.Optional[int] = None
    sample_rate: int = 16_000
    smoothing: typing.Optional[typing.Callable] = None

    @classmethod
    def finalize_dogmatic_config(cls, config):
        rate = config['sample_rate']
        config['smoothing'] = {
            'partial': conv_smoothing,
            'window_length': int(0.3 * rate),
            'threshold': int(0.1 * rate),
        }

    def __post_init__(self):
        if self.window_size is None:
            self.window_size = int(0.1 * self.sample_rate + 1)

        assert self.window_size % 2 == 1, self.window_size

    def __call__(self, example):
        if isinstance(example, dict):
            signal = example['audio_data']
            if signal.ndim == 2 and signal.shape[0] == 1:
                signal = signal[0]
            elif signal.ndim == 2 and signal.shape[0] != 1:
                raise ValueError(
                    'Only mono signals are supported but audio_data has shape '
                    f'{signal.shape}'
                )
            vad = super().__call__(signal)
            intervals = np.asarray(vad.intervals)
            start, stop = zip(*intervals)
            example['vad'] = vad
            example['vad_start_samples'] = start
            example['vad_stop_samples'] = stop
        else:
            example = super().__call__(example)
        return example

    def _detect_voice_activity(self, signal):
        assert signal.ndim == 1, signal.shape

        half_context = (self.window_size - 1) // 2
        std = np.std(signal)
        signal = signal - np.mean(signal)
        assert np.min(signal) < 0
        assert np.max(signal) > 0
        signal = np.abs(signal)

        sliding_max = np.max(pb.array.segment_axis(
            np.pad(signal, (half_context, half_context), mode='constant'),
            length=self.window_size, shift=1, axis=0, end='cut'
        ), axis=-1)

        assert sliding_max.shape == signal.shape, (
            sliding_max.shape, signal.shape
        )

        unconstrained = sliding_max > self.threshold * std

        return unconstrained

    def compute_vad(self, signal, time_resolution=True):
        assert time_resolution
        return pb.array.interval.ArrayInterval(
            self._detect_voice_activity(signal)
        )

    def vad_to_time(self, vad, time_length):
        assert time_length == vad.shape[-1], (time_length, vad.shape[-1])
        return vad