import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa
import torch


def plot_waveform(x, sr):
    """
    Plot the waveform

    Parameters
    ----------
    x : np.array
        waveform
    sr : int
        sampling rate

    Returns
    -------
    None
    """
    print(f'Signal shape: {x.shape}\n Sampling rate: {sr}\n')
    plt.figure(figsize=(16, 8))
    plt.plot(x, color='blue')
    plt.xlim([0, x.shape[0]])
    plt.xlabel('Time samples)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.tight_layout()
    plt.show()
    ipd.display(ipd.Audio(data=x, rate=sr))


def plot_selected_waveform(x, sr, start_stop):
    """
    Plot the waveform and highlight the selected sound of a bird.

    Parameters
    ----------
    x : np.array
        waveform
    sr : int
        sampling rate
    start_stop : List
        indices of subsequent chunks of filtered parts of waveform

    Returns
    -------
    None
    """
    plt.figure(figsize=(16, 8))
    plt.plot(x, color='blue')
    plt.xlim([0, x.shape[0]])
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')

    for i in range(1, len(start_stop), 2):
        plt.axvspan(start_stop[i - 1], start_stop[i], color='red', alpha=0.5)
    plt.grid()
    ipd.display(ipd.Audio(data=x, rate=sr))


def plot_torch_spectrogram(spectrogram, title=None, ylabel="freq_bin", ax=None):
    """
    Plot a spectrogram
    Parameters
    ----------
    spectrogram: torch.Tensor
        A 2D tensor to plot
    title: str
        Title of the plot
    ylabel: str
    ax: matplotlib.axes
        Optional axes to be used to plot

    Returns
    -------
    None
    """
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(spectrogram, origin="lower", aspect="auto", interpolation="nearest")


def plot_torch_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    """
    Plot a waveform
    Parameters
    ----------
    waveform: torch.Tensor
        1D waveform signal
    sample_rate: int
    title: str
    xlim: float
    ylim: float

    Returns
    -------
    None
    """
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c + 1}')
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show(block=False)
    ipd.display(ipd.Audio(data=waveform, rate=sample_rate))
