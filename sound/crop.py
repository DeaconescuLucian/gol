from pydub import AudioSegment


def crop_audio(input_file, output_file, start_ms, end_ms):
    audio = AudioSegment.from_file(input_file)

    # Crop the audio using array slicing
    cropped_audio = audio[start_ms:]

    # Export the cropped audio as a new file
    cropped_audio.export(output_file, format="wav")


if __name__ == "__main__":
    input_file = "auds/problem.mp3"  # Input audio file to be cropped
    output_file = "auds/bum.mp3"  # Output file with the cropped audio
    start_ms = 800  # Start position (in milliseconds) for the crop
    end_ms = 100  # End position (in milliseconds) for the crop

    crop_audio(input_file, output_file, start_ms, end_ms)