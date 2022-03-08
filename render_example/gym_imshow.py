import gym
import os
import matplotlib.pyplot as plt
from matplotlib import animation


def unit_render():
    """Just render an image"""
    env = gym.make('CartPole-v0')  # insert your favorite environment
    
    def render(i):
        plt.imshow(env.render(mode='rgb_array'))
        plt.savefig('frame' + str(i))
        return 
    
    env.reset()

    for i in range(1000):
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)
        render(i)
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


def video_render(episodes=200,
                 episodes_to_render=[23, 57],
                 how_much_episodes_to_render=5,
                 batch_episodes=100,
                 loop_iterations=500):
    # Make gym env
    env = gym.make('Pendulum-v1')

    def condition(n_episode):
        if (n_episode % batch_episodes) < how_much_episodes_to_render:
            return True
        return False
    os.makedirs('renders', exist_ok=True)
    # Run the env
    observation = env.reset()
    # frames_episodes = dict()
    for episode in range(episodes):
        if condition(n_episode=episode):
            frames = list()
        for step in range(loop_iterations):
            # Render to frames buffer
            if condition(n_episode=episode):
                frames.append(env.render(mode="rgb_array"))
            # Take random action
            action = env.action_space.sample()
            _, _, done, _ = env.step(action)
            print('Episode', episode, ', t =', step)
            # if done:
            #     print("Episode finished after {} timesteps".format(t + 1))
            #     break
        if condition(n_episode=episode):
            save_frames_as_mp4(frames,
                               path='renders',
                               filename='gym_animation_' + str(episode) + '.mp4')
            # frames_episodes.update({str(episode + 1): frames})
    env.close()

    # Save frames as mp4
    # for episode, frames in frames_episodes.items():
    #     save_frames_as_mp4(frames,
    #                        path='renders',
    #                        filename='gym_animation_' + episode + '.mp4')


if __name__ == '__main__':
    # unit_render()
    video_render()
