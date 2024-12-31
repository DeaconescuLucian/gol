from pydub import AudioSegment


def stretch_audio(input_file, output_file, desired_duration_ms):
    audio = AudioSegment.from_file(input_file)
    original_duration = len(audio)

    if original_duration >= desired_duration_ms:
        print("The audio is already longer than the desired duration.")
        return

    # Calculate the speedup factor needed to reach the desired duration
    speedup_factor = desired_duration_ms / original_duration

    # Stretch the audio to the desired duration
    stretched_audio = audio.speedup(playback_speed=speedup_factor)

    # Export the stretched audio as a new file
    stretched_audio.export(output_file, format="mp3")


if __name__ == "__main__":
    input_file = "auds/problem-high-cropped.mp3"  # Input audio file with a duration of 1.5 seconds
    output_file = "auds/problem-high-stretched.mp3"  # Output file with the audio stretched to 3 seconds
    desired_duration_ms = 3000  # Desired duration in milliseconds (3 seconds)

    stretch_audio(input_file, output_file, desired_duration_ms)