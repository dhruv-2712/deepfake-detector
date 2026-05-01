import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _linear_filterbank(n_filters: int, n_fft: int, sample_rate: int) -> torch.Tensor:
    """Triangular filterbank with linearly spaced centers. Returns (n_filters, n_fft//2+1)."""
    n_bins = n_fft // 2 + 1
    freq_bins = torch.linspace(0, sample_rate / 2, n_bins)
    centers = torch.linspace(0, sample_rate / 2, n_filters + 2)

    fb = torch.zeros(n_filters, n_bins)
    for m in range(n_filters):
        f_lo, f_mid, f_hi = centers[m], centers[m + 1], centers[m + 2]
        rise = (freq_bins >= f_lo) & (freq_bins <= f_mid)
        fall = (freq_bins > f_mid) & (freq_bins <= f_hi)
        fb[m, rise] = (freq_bins[rise] - f_lo) / (f_mid - f_lo + 1e-8)
        fb[m, fall] = (f_hi - freq_bins[fall]) / (f_hi - f_mid + 1e-8)
    return fb  # (n_filters, n_bins)


def _dct_matrix(n_filters: int, n_coeffs: int) -> torch.Tensor:
    """Type-II DCT matrix, shape (n_coeffs, n_filters)."""
    n = torch.arange(n_filters, dtype=torch.float32)
    k = torch.arange(n_coeffs, dtype=torch.float32).unsqueeze(1)
    return torch.cos(math.pi / n_filters * (n + 0.5) * k)


class AudioFeatureExtractor(nn.Module):
    """
    Extracts a (B, 512) embedding from raw waveforms using LFCC + LCNN.

    Pipeline:
      (B, T) waveform
        → STFT → 40-band linear filterbank → log energy → DCT[:20]
        → (B, 1, 20, T_frames)
        → 3-layer LCNN (Conv→MaxPool→SELU)
        → AdaptiveAvgPool2d(4,4) → Linear(1024, 512)
    """

    N_FFT = 512
    HOP_LENGTH = 160   # 10 ms at 16 kHz
    WIN_LENGTH = 400   # 25 ms at 16 kHz
    N_FILTERS = 40
    N_COEFFS = 20

    def __init__(self, sample_rate: int = 16000):
        super().__init__()
        self.sample_rate = sample_rate

        self.register_buffer("filterbank", _linear_filterbank(self.N_FILTERS, self.N_FFT, sample_rate))
        self.register_buffer("dct_mat", _dct_matrix(self.N_FILTERS, self.N_COEFFS))

        self.lcnn = nn.Sequential(
            # block 1
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.MaxPool2d(2, 2),
            nn.SELU(),
            # block 2
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(2, 2),
            nn.SELU(),
            # block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.SELU(),
            # pool + flatten
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.proj = nn.Linear(64 * 16, 512)

    def extract_lfcc(self, waveform: torch.Tensor) -> torch.Tensor:
        """(B, T) → (B, 1, N_COEFFS, T_frames)"""
        window = torch.hann_window(self.WIN_LENGTH, device=waveform.device)
        stft = torch.stft(
            waveform,
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH,
            win_length=self.WIN_LENGTH,
            window=window,
            return_complex=True,
        )  # (B, F, T_frames)

        power = stft.abs().pow(2)                              # (B, F, T_frames)
        energies = torch.matmul(self.filterbank, power)        # (B, N_FILTERS, T_frames)
        log_e = torch.log(energies + 1e-8)                     # (B, N_FILTERS, T_frames)
        lfcc = torch.matmul(self.dct_mat, log_e)               # (B, N_COEFFS, T_frames)
        return lfcc.unsqueeze(1)                               # (B, 1, N_COEFFS, T_frames)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (B, T) raw samples, float32
        feat = self.extract_lfcc(waveform)   # (B, 1, 20, T_frames)
        return self.proj(self.lcnn(feat))    # (B, 512)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioFeatureExtractor(sample_rate=16000).to(device)
    model.eval()

    dummy = torch.rand(2, 4 * 16000, device=device)  # batch=2, 4 seconds at 16kHz
    with torch.no_grad():
        out = model(dummy)
    print(f"Output shape: {out.shape}")  # expect torch.Size([2, 512])
