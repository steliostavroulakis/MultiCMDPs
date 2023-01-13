import random
from GRIDWORLD.gridworld_env import GridWorldEnv
import imageio

action_dict = dict()
action_dict[0] = 'UP'
action_dict[1] = 'DOWN'
action_dict[2] = 'LEFT'
action_dict[3] = 'RIGHT'

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

frames = []
env = GridWorldEnv()
obs = env.reset()
done = False
i = 0

while not done:
    env.render_gridworld_before(env.grid, env.pos,i)
    frames.append(imageio.imread('gridworld.png'))

    action = env.action_space.sample()  # select a random action
    #action = 2
    obs, reward, done, info = env.step(action)
    env.render_gridworld_after(env.grid, env.pos,i,action,reward)

    print(f'Action: {action_dict[action]}, Reward: {reward}, Done: {done}')
    frames.append(imageio.imread('gridworld.png'))
    i+=1

# create the video file from the list of frames
create_video(frames, 'gridworld.mp4', fps=0.5)