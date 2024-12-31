from pydub import AudioSegment


def combine_audio_files(input_file1, input_file2, output_file):
    audio1 = AudioSegment.from_file(input_file1)
    audio2 = AudioSegment.from_file(input_file2)

    combined_audio = audio1 + audio2

    # Export the combined audio as a new file
    combined_audio.export(output_file, format="mp3")


def overlay_audio(input_file1, input_file2, output_file, position_ms=0):
    audio1 = AudioSegment.from_file(input_file1)
    audio2 = AudioSegment.from_file(input_file2)

    # Adjust the position (in milliseconds) to overlay audio1 on audio2
    combined_audio = audio2.overlay(audio1, position=position_ms)

    # Export the combined audio as a new file
    combined_audio.export(output_file, format="mp3")


def repeat(input_file1, times, output_file):
    audio1 = AudioSegment.from_file(input_file1)

    combined_audio = audio1 + audio1

    for i in range(times):
        combined_audio += audio1

    # Export the combined audio as a new file
    combined_audio.export(output_file, format="mp3")


def repeat2(input_file1, input_file2, times, output_file):
    audio1 = AudioSegment.from_file(input_file1)
    audio2 = AudioSegment.from_file(input_file2)
    combined_audio = audio1 + audio2

    for i in range(times):
        combined_audio += audio1
        combined_audio += audio2

    # Export the combined audio as a new file
    combined_audio.export(output_file, format="mp3")

if __name__ == "__main__":
    input_file1 = "auds/tz.mp3"
    input_file2 = "auds/hb.mp3"
    output_file = "auds/ciorba.mp3"

    overlay_audio(input_file1, input_file2, output_file)

# if __name__ == "__main__":
#     input_file1 = "auds/tz.mp3"
#     output_file = "auds/tz2.mp3"
#     repeat(input_file1, 1, output_file)

# if __name__ == "__main__":
#     input_file1 = "auds/hb/hb.mp3"
#     input_file2 = "auds/hb/hb.mp3"
#     output_file = "auds/hb.mp3"
#     repeat2(input_file1,input_file2, 5, output_file)