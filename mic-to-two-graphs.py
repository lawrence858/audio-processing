import pyaudio
import numpy as np
import matplotlib.pyplot as plt

plt.ion()

CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 22050
CHUNKS_PER_INTERVAL = 8

samples_per_interval = CHUNK_SIZE * CHUNKS_PER_INTERVAL

sample_numbers = np.arange(samples_per_interval)
duration = samples_per_interval / RATE
x_freq = sample_numbers / duration
x_freq = x_freq[range(samples_per_interval // 2)]

print("samples_per_interval: {}".format(samples_per_interval))
print("ms per interval: {:.0f}".format(1000 * duration))

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

axs[0].set_title('Sound Wave')
axs[0].set_ylim(-2 ** 15, 2 ** 15)  # full range should be -2**15 .. 2**15
x = np.arange(0, samples_per_interval)
y = np.zeros(samples_per_interval)
audio_line, = axs[0].plot(x, y)

axs[1].set_title('Spectrum')
axs[1].set_ylim(0, 50)
spectrum_line, = axs[1].plot(x_freq, y[range(samples_per_interval // 2)])


def record_audio():
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=1,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)
    frames = []

    for i in range(0, CHUNKS_PER_INTERVAL):
        data = stream.read(CHUNK_SIZE)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    return b''.join(frames)


def get_abs_fft(audio):
    audio_samples = np.frombuffer(audio, dtype=np.int16)
    num_samples = len(audio_samples)
    fft_values = np.fft.fft(audio_samples)
    fft_values_scaled = fft_values / num_samples
    fft_values_scaled = fft_values_scaled[range(num_samples // 2)]
    return abs(fft_values_scaled)


def main():
    while True:
        audio = record_audio()
        audio_values = np.frombuffer(audio, dtype=np.int16)
        freq_amplitudes = get_abs_fft(audio)
        audio_line.set_ydata(audio_values)
        spectrum_line.set_ydata(freq_amplitudes)

        fig.canvas.draw()  # Redraw the plot
        fig.canvas.flush_events()  # flush the GUI events (optional)
        plt.pause(0.0001)  # Pause briefly to allow the plot to update (needed)
        get_abs_fft(audio)


if __name__ == '__main__':
    main()
