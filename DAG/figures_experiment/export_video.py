import imageio

def create_video(frames, out_filename, fps=1):
    """
    Create a video file from a sequence of images.
    
    Args:
        frames: a list of NumPy arrays representing the frames of the video
        out_filename: the name of the output video file
        fps: the frame rate of the video (frames per second)
    """
    writer = imageio.get_writer(out_filename, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

iterates = [39,35,31,28,25,23,21,19,17,15,14,12,11,10,9,8]
frames = []

for i in iterates:
    frames.append(imageio.imread(f'experiment_result_{i}.png'))

create_video(frames, 'experiment.mp4')