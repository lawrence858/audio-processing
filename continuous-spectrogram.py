import pyaudio
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import signal

CHUNK_SIZE = 4096
FORMAT = pyaudio.paFloat32
RATE = 22050
CHUNKS_PER_INTERVAL = 2
SAMPLES_PER_INTERVAL = CHUNKS_PER_INTERVAL * CHUNK_SIZE

global py_audio
global stream


def kill_handler(*args):
    close_audio()
    sys.exit(0)


def open_audio():
    global py_audio
    global stream
    py_audio = pyaudio.PyAudio()
    stream = py_audio.open(format=FORMAT,
                           channels=1,
                           rate=RATE,
                           input=True,
                           frames_per_buffer=CHUNK_SIZE)


def close_audio():
    stream.stop_stream()
    stream.close()
    py_audio.terminate()


def record_audio():
    frames = []
    for i in range(0, CHUNKS_PER_INTERVAL):
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        frames.append(data)
    binary_values = b''.join(frames)
    return binary_values


def show_spectrogram(samples):
    updated_spectrogram = librosa.feature.melspectrogram(y=samples, sr=RATE)
    updated_spectrogram_db = librosa.power_to_db(updated_spectrogram, ref=np.max)
    librosa.display.specshow(updated_spectrogram_db, x_axis='time', y_axis='mel', sr=RATE, ax=ax)


def update(frame):
    global current_location
    audio = record_audio()
    audio_values = np.frombuffer(audio, dtype=np.float32)

    if current_location >= len(sound_buffer):
        # shift left
        current_location = len(sound_buffer) - len(audio_values)
        sound_buffer[0:current_location] = sound_buffer[len(audio_values):current_location + len(audio_values)]

    # insert new samples
    sound_buffer[current_location:current_location + len(audio_values)] = audio_values
    show_spectrogram(sound_buffer)

    current_location = current_location + len(audio_values)


if __name__ == '__main__':
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
    ax.set_title('Spectrogram')

    # allocate room for 10 seconds rounded down to multiple of SAMPLES_PER_INTERVAL
    sound_buffer = np.zeros(SAMPLES_PER_INTERVAL * ((RATE * 10) // SAMPLES_PER_INTERVAL))
    current_location = 0

    signal.signal(signal.SIGINT, kill_handler)
    open_audio()
    anim = FuncAnimation(fig, update, frames=1, repeat=True)
    plt.show()
