from pydub import AudioSegment


def reverse_audio(input_file, output_file):
    audio = AudioSegment.from_file(input_file)

    # Reverse the audio
    reversed_audio = audio.reverse()

    # Export the reversed audio as a new file
    reversed_audio.export(output_file, format="wav")


if __name__ == "__main__":
    input_file = "auds/drums/drum-percussion.wav"  # Input audio file to be reversed
    output_file = "auds/drums/drum-percussion-reversed.wav"  # Output file with the audio reversed

    reverse_audio(input_file, output_file)