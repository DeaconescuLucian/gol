from pydub import AudioSegment


def change_pitch(input_file, output_file, semitones=1):
    AudioSegment.converter = "D:/ffmpeg/ffmpeg-master-latest-win64-gpl/bin/ffmpeg"
    audio = AudioSegment.from_file(input_file)

    # Shift the pitch by the specified number of semitones
    new_audio = audio._spawn(audio.raw_data, overrides={
        "frame_rate": int(audio.frame_rate / (2.0 ** (semitones / 12.0)))
    })

    # Export the modified audio as a new file
    new_audio.export(output_file, format="mp3")


if __name__ == "__main__":
    input_file = "auds/tz-space.mp3"
    output_file = "auds/tzs.mp3"
    semitones_higher = 5  # Adjust this value to set how much higher you want the audio (e.g., 1 for one semitone higher)

    change_pitch(input_file, output_file, semitones=semitones_higher)