import gym
import os
import matplotlib.pyplot as plt
from matplotlib import animation


def unit_render():
    """Just render an image"""
    env = gym.make('CartPole-v0')  # insert your favorite environment
    def render(): return plt.imshow(env.render(mode='rgb_array'))
    env.reset()
    render()


def save_frames_as_mp4(frames, path='./', filename='gym_animation.mp4'):
    """
    Save frames from render as mp4 using ffmpeg.
    """
    # Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0,
               frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(
        plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(os.path.join(path, filename),
              writer='ffmpeg',
              fps=60)


def video_render(episodes=100,
                 episodes_to_render=[23, 57],
                 loop_iterations=500):
    # Make gym env
    env = gym.make('Pendulum-v1')

    # Run the env
    observation = env.reset()
    frames_episodes = dict()
    for episode in range(episodes):
        if episode + 1 in episodes_to_render:
            frames = list()
        for t in range(loop_iterations):
            # Render to frames buffer
            if episode + 1 in episodes_to_render:
                frames.append(env.render(mode="rgb_array"))
            # Take random action
            action = env.action_space.sample()
            _, _, done, _ = env.step(action)
            print('Episode', episode, ', t =', t)
            # if done:
            #     print("Episode finished after {} timesteps".format(t + 1))
            #     break
        if episode + 1 in episodes_to_render:
            frames_episodes.update({str(episode + 1): frames})
    env.close()

    # Save frames as mp4
    os.makedirs('renders', exist_ok=True)
    for episode, frames in frames_episodes.items():
        save_frames_as_mp4(frames,
                           path='renders',
                           filename='gym_animation_' + episode + '.mp4')


if __name__ == '__main__':
    unit_render()
    video_render()
