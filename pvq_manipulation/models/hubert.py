import os
from pathlib import Path
from contextlib import nullcontext
import typing as tp
from typing import List, Tuple, Optional

import einops
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

import padertorch as pt
from padertorch.contrib.je.modules.conv_utils import (
    compute_conv_output_sequence_lengths
)
from padertorch.utils import to_numpy
from transformers.models.hubert.modeling_hubert import HubertModel

# See https://ieeexplore.ieee.org/abstract/document/9814838, Fig. 2
PR_BASE_LAYER = 11
PR_LARGE_LAYER = 22
SID_BASE_LAYER = 4
SID_LARGE_LAYER = 6


def tuple_to_int(sequence) -> list:
    return list(map(lambda t: t[0], sequence))


class HubertExtractor(pt.Module):
    """Extract HuBERT features from raw waveform.

    Args:
        model_name (str): Name of the pretrained HuBERT model on huggingface.co.
            Defaults to "facebook/hubert-large-ll60k".
        layer (int): Index of the layer to extract features from. Defaults to
            22.
        freeze (bool): If True, freeze the weights of the encoder
            (i.e., no finetuning of Transformer layers). Defaults to True.
    """

    def __init__(
            self,
            model_name: str = "facebook/hubert-large-ll60k",
            layer: tp.Union[int, str] = PR_LARGE_LAYER,
            freeze: bool = True,
            detach: bool = False,
            device: str = "cpu",
            backend: str = "torchaudio",
            storage_dir: str = None,
    ):
        super().__init__()

        if not freeze and detach:
            raise ValueError(
                'detach=True only supported if freeze=True\n'
                f'Got: freeze={freeze}, detach={detach}'
            )
        if backend == "torchaudio":
            bundle = getattr(torchaudio.pipelines, model_name)
            self.model = bundle.get_model(dl_kwargs={'model_dir': storage_dir}).eval().to(device)
            self.sampling_rate = bundle.sample_rate
        else:
            raise ValueError(f'Unknown backend: {backend}')
        self.backend = backend

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            # Always freeze feature extractor and feature projection layers
            for param in self.model.feature_extractor.parameters():
                param.requires_grad = False
            for param in self.model.feature_projection.parameters():
                param.requires_grad = False

        self.layer = layer
        self.freeze = freeze
        self.detach = detach

        if self.layer == 'all':
            self.weights = torch.nn.Parameter(
                torch.ones(24), requires_grad=True
            )

    @property
    def cache_dir(self):
        return Path(os.environ['STORAGE_ROOT']) / 'huggingface' / 'hub'

    @property
    def context(self):
        if self.detach:
            return torch.no_grad()
        else:
            return nullcontext()

    def compute_output_lengths(
            self, input_lengths: Optional[List[int]]
    ) -> Optional[List[int]]:
        """Compute the number of time frames for each batch entry.

        Args:
            input_lengths: List with number of samples per batch entry.

        Returns:
            List with number of time frames per batch entry.
        """
        if input_lengths is None:
            return input_lengths
        output_lengths = np.asarray(input_lengths) + self.window_size - 1
        for kernel_size, dilation, stride in zip(
                self.kernel_sizes, self.dilations, self.strides,
        ):
            output_lengths = compute_conv_output_sequence_lengths(
                output_lengths,
                kernel_size=kernel_size,
                dilation=dilation,
                pad_type=None,
                stride=stride,
            )
        return output_lengths.tolist()

    def forward(
        self,
        time_signal: torch.Tensor,
        sampling_rate: int,
        sequence_lengths: Optional[List[int]] = None,
        extract_features: bool = False,
        other_inputs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[List[int]]]:
        """Extract HuBERT features from raw waveform.

        Args:
            time_signal: Time signal of shape (batch, 1, time) or (batch, time)
                sampled at 16 kHz.
            sequence_lengths: List with number of samples per batch entry.

        Returns:
            x (torch.Tensor): HuBERT features of shape
                (batch, D, time frames).
            sequence_lengths (List[int]): List with number of time frames per
                batch entry.
        """
        del other_inputs

        if time_signal.ndim == 3:
            time_signal = einops.rearrange(time_signal, 'b c t -> (b c) t')

        time_signal = torchaudio.functional.resample(
            time_signal, sampling_rate, self.sampling_rate
        )
        if sequence_lengths is not None:
            if isinstance(sequence_lengths, (list, tuple)):
                sequence_lengths = torch.tensor(sequence_lengths).long() \
                    .to(time_signal.device)
            sequence_lengths = (
                    sequence_lengths / sampling_rate * self.sampling_rate
            ).long()

        if self.freeze or self.detach:
            self.model.eval()
        with self.context:
            if self.backend == "torchaudio":
                self.model: torchaudio.models.Wav2Vec2Model
                x, sequence_lengths = self.model.extract_features(
                    time_signal, lengths=sequence_lengths,
                    num_layers=self.layer,
                )
                if isinstance(self.layer, int):
                    x = x[-1].transpose(1, 2)
                else:
                    raise NotImplementedError(self.layer)
                return x, sequence_lengths

            self.model: HubertModel
            n_pad = self.window_size - 1
            time_signal = F.pad(time_signal, (0, n_pad), value=0)
            if extract_features:
                features = self.model.feature_extractor(time_signal.float()) \
                    .transpose(1, 2)
                x = self.model.feature_projection(features).transpose(1, 2)
            else:
                outputs = self.model(
                    time_signal.float(), output_hidden_states=True
                )
                if isinstance(self.layer, int):
                    x = outputs.hidden_states[self.layer].transpose(1, 2)
                    if self.detach:
                        x = x.detach()
                elif self.layer == 'all':
                    hidden_states = []
                    for _, hidden_state in enumerate(outputs.hidden_states):
                        x = hidden_state.transpose(1, 2)
                        if self.detach:
                            x = x.detach()
                        hidden_states.append(x)
                    hidden_states = torch.stack(hidden_states)  # (L, B, D, T)
                    x = (hidden_states * self.weights[:, None, None, None]) \
                        .sum(dim=0)
                else:
                    raise ValueError(f'Unknown layer: {self.layer}')

        sequence_lengths = to_numpy(sequence_lengths)
        sequence_lengths = self.compute_output_lengths(sequence_lengths)

        return x.unsqueeze(1), sequence_lengths
