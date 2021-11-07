#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from deep_rl import *

def dqn_pixel(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game,)
    config.eval_env = config.task_fn()

    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.00025/4, eps=1.5e-04)
    config.network_fn = lambda: VanillaNet(config.action_dim, NatureConvBody(in_channels=config.history_length))
    config.random_action_prob = LinearSchedule(1.0, 0.1, 1e6)
    config.batch_size = 32
    config.discount = 0.99
    config.history_length = 4
    config.max_steps = int(2e7)
    replay_kwargs = dict(
        memory_size=int(1e6),
        batch_size=config.batch_size,
        n_step=config.n_step,
        discount=config.discount,
        history_length=config.history_length,
    )
    config.replay_fn = lambda: ReplayWrapper(config.replay_cls, replay_kwargs, config.async_replay)
    config.replay_eps = 0.01
    config.replay_alpha = 0.5
    config.replay_beta = LinearSchedule(0.4, 1.0, config.max_steps)

    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.target_network_update_freq = 10000
    config.exploration_steps = 20000
    config.sgd_update_frequency = 4
    config.gradient_clip = 10
    config.double_q = True
    config.async_actor = False
    config.eval_only = False
    config.eval_episodes = 100
    config.save_interval = 2e6
    run_steps(DQNAgent(config))

def bdqn_pixel(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()

    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.0025, eps=1.5e-04)
    config.network_fn = lambda: BDQNNet(NatureConvBody(in_channels=config.history_length))
    config.batch_size = 256
    config.discount = 0.99
    config.history_length = 4
    config.max_steps = int(2e7)
    replay_kwargs = dict(
        memory_size=int(1e6),
        batch_size=config.batch_size,
        n_step=config.n_step,
        discount=config.discount,
        history_length=config.history_length,
    )
    config.replay_fn = lambda: ReplayWrapper(config.replay_cls, replay_kwargs, config.async_replay)
    config.replay_eps = 0.01
    config.replay_alpha = 0.5
    config.replay_beta = LinearSchedule(0.4, 1.0, config.max_steps)

    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.target_network_update_freq = 10000
    config.exploration_steps = 20000
    config.sgd_update_frequency = 4
    config.thompson_sampling_freq = 1000
    config.bdqn_learn_frequency = 50000
    config.prior_var = 0.001
    config.noise_var = 1
    config.var_k = LinearSchedule(1e-2, 1e-4, config.max_steps)
    config.gradient_clip = 10
    config.double_q = True
    config.async_actor = False
    config.eval_only = True
    run_steps(BDQNAgent(config))

if __name__ == '__main__':
    print(torch.__version__)
    mkdir('log')
    mkdir('tf_log')
    set_one_thread()
    random_seed()
    # -1 is CPU, a positive integer is the index of GPU
    # select_device(-1)
    select_device(0)
    

    game = 'BreakoutNoFrameskip-v4'
    dqn_pixel(game=game, n_step=3, replay_cls=PrioritizedReplay, async_replay=True, run=0, remark='dqn32adam')
#     bdqn_pixel(game=game, n_step=3, replay_cls=PrioritizedReplay, async_replay=True)
