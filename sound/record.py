import pyaudio
import wave


def record_audio(output_file, duration=5, sample_rate=44100, channels=2, chunk_size=1024):
    audio_format = pyaudio.paInt16
    audio_frames = []

    p = pyaudio.PyAudio()

    stream = p.open(format=audio_format,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk_size)

    print("Recording audio...")

    for i in range(0, int(sample_rate / chunk_size * duration)):
        data = stream.read(chunk_size)
        audio_frames.append(data)

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(audio_format))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(audio_frames))


if __name__ == "__main__":
    output_file = "auds/problem.mp3"
    recording_duration = 1  # Recording duration in seconds

    record_audio(output_file, duration=recording_duration)
