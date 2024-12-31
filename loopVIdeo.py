from moviepy.editor import VideoFileClip, concatenate_videoclips

# Load the video file
video_clip = VideoFileClip("D:\\TF\\vids\\mini_sat.mp4")

# Specify the number of times to loop the video
num_loops = 10

# Create a list to hold copies of the video clip
video_clips = [video_clip for _ in range(num_loops)]

# Concatenate the video clips to create the looped video
looped_clip = concatenate_videoclips(video_clips)

# Export the looped video with sound
looped_clip.write_videofile("vids/mini_sat_loop.mp4", audio=True)

# Close the video clips
video_clip.close()
looped_clip.close()