import time
import gym
import gym.spaces
import warnings
import cv2
import torch
import torch.nn as nn  # neural network package
import torch.optim as optim  # optimization package
import numpy as np
import collections
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import datetime


class PressFireButton(gym.Wrapper):
    def __init__(self, env=None):
        super(PressFireButton, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs


class Observe(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(Observe, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        totalReward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            totalReward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, totalReward, done, info

    def reset(self):
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ScaleDownFrames(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ScaleDownFrames, self).__init__(env)
        self.observationSpace = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ScaleDownFrames.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class StackFramesForGameDynamics(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(StackFramesForGameDynamics, self).__init__(env)
        self.dtype = dtype
        oldSpace = env.observationSpace
        self.observationSpace = gym.spaces.Box(oldSpace.low.repeat(n_steps, axis=0),
                                               oldSpace.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observationSpace.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


class GetTorchFromImage(gym.ObservationWrapper):
    def __init__(self, env):
        super(GetTorchFromImage, self).__init__(env)
        oldShape = self.observationSpace.shape
        self.observationSpace = gym.spaces.Box(low=0.0, high=1.0, shape=(oldShape[-1],
                                                                         oldShape[0], oldShape[1]), dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaleFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.divide(np.array(obs).astype(np.float32), 255.0)


def makeEnv(env_name):
    env = gym.make(env_name)
    env = Observe(env)
    env = PressFireButton(env)
    env = ScaleDownFrames(env)
    env = GetTorchFromImage(env)
    env = StackFramesForGameDynamics(env, 4)
    return ScaleFrame(env)


class DeepNN(nn.Module):
    def __init__(self, observationSpaceShape, n_actions):
        super(DeepNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(observationSpaceShape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        convOutSize = self._getConvOut(observationSpaceShape)
        self.fc = nn.Sequential(
            nn.Linear(convOutSize, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _getConvOut(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


class ExpReplay:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batchSize):
        indices = np.random.choice(len(self.buffer), batchSize, replace=False)
        states, actions, rewards, dones, nextStates = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(nextStates)


class Agent:
    def __init__(self, env, expBuffer):
        self.env = env
        self.expBuffer = expBuffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.totalReward = 0.0

    def playStep(self, net, epsilon=0.0, device="cpu"):

        doneReward = None
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            stateA = np.array([self.state], copy=False)
            stateV = torch.tensor(stateA).to(device)
            qValsV = net(stateV)
            _, actV = torch.max(qValsV, dim=1)
            action = int(actV.item())

        newState, reward, isDone, _ = self.env.step(action)
        self.totalReward += reward

        exp = Experience(self.state, action, reward, isDone, newState)
        self.expBuffer.append(exp)
        self.state = newState
        if isDone:
            doneReward = self.totalReward
            self._reset()
        return doneReward


# function to plot a line graph with gaussian normalisation
def graph(items, labelX, labelY, labelLine, title):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(labelX)
    ax.set_ylabel(labelY)
    sig = 6
    for i in range(0, len(items), 100):
        sig = int(sig * 1.4)
    ax.plot(gaussian_filter1d(items, sigma=sig), label=labelLine)
    ax.legend()
    plt.show()


def testEnvironment(envName, device):
    testEnv = makeEnv(envName)
    testNet = DeepNN(testEnv.observationSpace.shape, testEnv.action_space.n).to(device)
    return testNet


if __name__ == "__main__":
    print(">>>Training starts at ", datetime.datetime.now())

    # set up
    envName = "PongNoFrameskip-v4"
    warnings.filterwarnings('ignore')

    # CHANGE to "cpu" if you do not have cuda enabled system !!!!
    device = torch.device("cuda")

    # test env
    print(testEnvironment(envName, device))

    # initialise
    mean_reward_bound = 19.0
    gamma = 0.99
    batchSize = 32
    replaySize = 150000
    lr = 1e-4
    syncTargetFrames = 10
    replayStartSize = 150000
    epsStart = 1.0
    epsDecay = 0.999985
    epsMin = 0.02
    Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

    # make env and NNs for training
    env = makeEnv(envName)
    net = DeepNN(env.observationSpace.shape, env.action_space.n).to(device)
    targetNet = DeepNN(env.observationSpace.shape, env.action_space.n).to(device)
    buffer = ExpReplay(replaySize)
    agent = Agent(env, buffer)
    epsilon = epsStart
    optimizer = optim.Adam(net.parameters(), lr=lr)
    totalRewards, allRewards = [], []
    bestAverageReward = None
    frameId = 0

    while 1:
        frameId += 1
        epsilon = max(epsilon * epsDecay, epsMin)

        reward = agent.playStep(net, epsilon, device=device)
        if reward is not None:
            totalRewards.append(reward)
            averageReward = np.mean(totalRewards[-100:])
            allRewards.append(averageReward)
            if len(totalRewards) % 100 == 0:
                graph(allRewards, 'games', 'rewards', 'rewards', 'Average of last 100 rewards per game')

            print("%d:  %d games, mean reward %.3f, (epsilon %.2f)" % (frameId, len(totalRewards),
                                                                       averageReward, epsilon))
            if bestAverageReward is None or bestAverageReward < averageReward:
                torch.save(net.state_dict(), envName + "-best.dat")
                bestAverageReward = averageReward
                if bestAverageReward is not None:
                    print("Best mean reward updated %.3f" % bestAverageReward)

            if averageReward > mean_reward_bound:
                print("Solved in %d Atari frames!" % frameId)
                break

        if len(buffer) < replayStartSize:
            continue

        # sample mini-batch of transactions from replay memory
        batch = buffer.sample(batchSize)
        states, actions, rewards, dones, nextStates = batch

        # wrap numpy arrays with batch data in torch tensors
        statesV = torch.tensor(states).to(device)
        nextStatesV = torch.tensor(nextStates).to(device)
        actionsV = torch.tensor(actions, dtype=torch.int64).to(device)
        rewardsV = torch.tensor(rewards).to(device)
        doneMask = torch.ByteTensor(dones).to(device)

        # pass observations to first model and extract the specific Q-values
        stateActionValues = torch.gather(net(statesV), 1, actionsV.unsqueeze(-1)).squeeze(-1)

        # apply the target network to our next state observations and calculate the max q-value along action dimension 1
        nextStateValues = targetNet(nextStatesV).max(1)[0]

        # if transition in batch is from last step in the episode then
        # action value does not have a discounted reward of next state
        nextStateValues[doneMask] = 0.0

        # makes copy without connection to parents operation to
        # prevent gradients flowing into the target network's graph
        nextStateValues = nextStateValues.detach()

        # calculate Bellman approximation value for the vector of the expected state-action value for every transition
        # in the replay buffer
        expectedStateActionValues = nextStateValues * gamma + rewardsV
        # print('expected_state_action_values:', expected_state_action_values)

        # calculate mean squared error loss
        lossT = nn.MSELoss()(stateActionValues, expectedStateActionValues)

        # update the neural network using Adam optimizer SGD
        optimizer.zero_grad()
        lossT.backward()
        optimizer.step()

        # sync params from main network to the target network every sync_target_frames
        if len(totalRewards) % syncTargetFrames == 0:
            targetNet.load_state_dict(net.state_dict())

    print(">>>Training ends at ", datetime.datetime.now())
