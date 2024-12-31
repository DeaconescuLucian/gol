from moviepy.editor import VideoFileClip


def separate_audio_and_video(input_video_path, output_audio_path, output_video_path):
    # Load the video clip
    video_clip = VideoFileClip(input_video_path)

    # Extract the audio
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(output_audio_path)

    # Extract the video
    video_clip = video_clip.set_audio(None)
    video_clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac')


if __name__ == "__main__":
    # Replace these paths with your actual input and output paths
    input_video_path = "input_video.mp4"
    output_audio_path = "auds/output_audio.mp3"
    output_video_path = "vids/output_video.mp4"

    separate_audio_and_video(input_video_path, output_audio_path, output_video_path)