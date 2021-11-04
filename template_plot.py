import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
# plt.rc('text', usetex=True)
from deep_rl.utils import plot

def plot_atari():
    plotter = plot.Plotter()
    games = [
        'BreakoutNoFrameskip-v4',
    ]

    patterns = [
        # 'remark_a2c',
        'doubledqnprioritized3',
        'doublebdqnprioritized3',
        # 'remark_n_step_dqn',
        # 'remark_option_critic',
        # 'remark_quantile',
        # 'remark_ppo',
        # 'remark_rainbow',
    ]

    labels = [
        'DQN',
        'BDQN'
        # 'A2C',
        # 'C51',
        # 'DQN',
        # 'N-Step DQN',
        # 'OC',
        # 'QR-DQN',
        # 'PPO',
        # 'Rainbow'
    ]

    plotter.plot_games(games=games,
                       patterns=patterns,
                       agg='mean',
                       downsample=100,
                       labels=labels,
                       right_align=False,
                       tag=plotter.RETURN_TRAIN,
                       root='./data/atari',
                       interpolation=0,
                       window=10,
                       )

    # plt.show()
    plt.tight_layout()
    plt.savefig('images/Breakout-dqn.png', bbox_inches='tight')


if __name__ == '__main__':
    # mkdir('images')
    # plot_ppo()
    # plot_ddpg_td3()
    plot_atari()