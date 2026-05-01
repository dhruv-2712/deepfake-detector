from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


class ASVspoofDataset(Dataset):
    """
    ASVspoof 2019 LA dataset loader.

    Protocol file format (space-separated):
        col 0: speaker ID
        col 1: utterance ID
        col 2: -
        col 3: system ID (or '-' for bonafide)
        col 4: label  ('bonafide' = 0, 'spoof' = 1)

    Audio files are expected at:
        audio_dir/{utterance_id}.flac  (or .wav)

    Returns: (waveform_tensor, label)
        waveform_tensor: (T,) float32, T = 4 * sample_rate = 64000
    """

    SAMPLE_RATE = 16000
    DURATION = 4.0
    NUM_SAMPLES = int(DURATION * SAMPLE_RATE)  # 64000

    def __init__(self, audio_dir: str, protocol_file: str):
        self.audio_dir = Path(audio_dir)
        self.samples: List[Tuple[str, int]] = []
        self._parse_protocol(protocol_file)

    def _parse_protocol(self, protocol_file: str):
        with open(protocol_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                utt_id = parts[1]
                label = 0 if parts[4].lower() == "bonafide" else 1
                self.samples.append((utt_id, label))

    def _load_audio(self, utt_id: str) -> torch.Tensor:
        for ext in (".flac", ".wav", ".mp3"):
            path = self.audio_dir / (utt_id + ext)
            if path.exists():
                break
        else:
            # Return silence if file is missing (graceful degradation during dev)
            return torch.zeros(self.NUM_SAMPLES)

        waveform, _ = librosa.load(
            str(path),
            sr=self.SAMPLE_RATE,
            mono=True,
            duration=self.DURATION,
        )
        waveform = librosa.util.fix_length(waveform, size=self.NUM_SAMPLES)
        return torch.from_numpy(waveform.astype(np.float32))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        utt_id, label = self.samples[idx]
        waveform = self._load_audio(utt_id)
        return waveform, label
