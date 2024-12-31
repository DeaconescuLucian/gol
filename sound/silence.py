from pydub import AudioSegment


def multiply(sound, times):
    final_sound = sound
    for i in range(times-1):
        final_sound += sound
    return final_sound

def add_silence(input_file, input_file1, output_file, times, silence_duration_ms=250):
    silence = AudioSegment.silent(duration=silence_duration_ms)
    original_audio = AudioSegment.from_file(input_file)
    original_audio1 = AudioSegment.from_file(input_file1)
    combined_audio = original_audio1 + multiply(silence, times)
    for i in range(10):
        combined_audio += original_audio1 + multiply(silence, times - i)

    combined_audio += original_audio + silence + original_audio + original_audio1
    for i in range(times):
        combined_audio += original_audio + silence + original_audio + original_audio1

    combined_audio.export(output_file, format="mp3")


if __name__ == "__main__":
    input_file = "auds/drums/tz.wav"
    input_file1 = "auds/bum.mp3"
    output_file = "auds/ciorba.mp3"
    silence_duration_ms = 25

    add_silence(input_file, input_file1, output_file, 10, silence_duration_ms)