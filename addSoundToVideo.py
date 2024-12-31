from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_audioclips

# Paths to your files
video_file = 'D:\\TF\\vids\\time-warp60fps.mp4'
audio_file = 'D:\\TF\\sounds\\dark.mp3'

# Load video clip and audio clip
video_clip = VideoFileClip(video_file)
audio_clip = AudioFileClip(audio_file)

# Ensure audio duration matches video duration (looping if necessary)
if audio_clip.duration < video_clip.duration:
    # Calculate how many times to repeat the audio
    loops_required = int(video_clip.duration // audio_clip.duration) + 1
    audio_clips = [audio_clip] * loops_required
    looped_audio_clip = concatenate_audioclips(audio_clips)

# Set the audio of the video clip
video_clip = video_clip.set_audio(looped_audio_clip.subclip(0, video_clip.duration))

# Export the final video with the background sound
output_file = 'D:\\TF\\vids\\time-warp-sound.mp4'
video_clip.write_videofile(output_file, codec='libx264', audio_codec='aac')

# Close the clips to free up resources
video_clip.close()
audio_clip.close()