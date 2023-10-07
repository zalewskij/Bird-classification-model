import matplotlib.pyplot as plt
import IPython.display as ipd


def plot_waveform(x, sr):
    """
    Plot the waveform

    Parameters
    ----------
    x : np.array
        waveform
    sr : int
        sampling rate
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


def plot_selected_waveform(x, start_stop):
    """
    Plot the waveform,

    Parameters
    ----------
    x : np.array
        waveform
    start_stop : List
        indices of subsequent chunks of filtered parts of waveform
    """
    plt.figure(figsize=(16, 8))
    plt.plot(x, color='blue')
    plt.xlim([0, x.shape[0]])
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')

    for i in range(0, len(start_stop), 2):
        plt.axvspan(start_stop[i], start_stop[i + 1], color='red', alpha=0.5)
    plt.grid()
