import gym
import gym_cartpole_swingup
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
#shows animation at the end of each policy update
env = gym.make("CartPoleSwingUp-v0")

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)
    
def show(weights, nn, index, repeat=1):#w1_,w2_,w3_,w4_,
    for each_game in range(repeat):
        prev_obs = []
        done = False
        env.reset()
        frames = []
        while not done:
            env.render()
            if len(prev_obs) == 0:
                action = env.action_space.sample()
            else:
                prev_obs = torch.tensor(new_observation,dtype=torch.float32).reshape([1, 5])
                m = nn.forward(prev_obs, weights, index)
                action = np.array(torch.normal(mean=float(m), std=torch.tensor(0.)))
            new_observation, reward, done, info = env.step(action)
            frames.append(env.render(mode="rgb_array"))
            #print(done)
            #print(info)
            prev_obs = new_observation
        env.close()
        save_frames_as_gif(frames)

def diagramm(T,m,upper,lower):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    m = np.array(m)
    upper = np.array(upper)
    lower = np.array(lower)
    np.save("m.npy",m)
    np.save("upper.npy",upper)
    np.save("lower.npy",lower)
    ax1.plot(range(0, T-1), m)
    plt.show()
    return
