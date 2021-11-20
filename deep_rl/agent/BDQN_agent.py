#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from ..utils import *
import time
from .BaseAgent import *
from .DQN_agent import DQNActor

class BDQNActor(BaseActor):
    def __init__(self, config):
        BaseActor.__init__(self, config)
        self.config = config
        self.current_action = 0
        self.sampled_mean = 0
        self.start()

    def compute_q(self, prediction):
        q_values = to_np(prediction['q'])
        return q_values
    
    def update_mean(self, sampled_mean):
        self.sampled_mean = sampled_mean

    def _transition(self):
        if self._state is None:
            self._state = self._task.reset()
        config = self.config
        with config.lock and torch.no_grad():
            prediction = self._network(config.state_normalizer(self._state))
        q_values = prediction['q']
        action = to_np(torch.argmax(torch.matmul(q_values, self.sampled_mean.T), dim=-1))
        next_state, reward, done, info = self._task.step(action)
        entry = [self._state, action, reward, next_state, done, info]
        self._total_steps += 1
        self._state = next_state
        return entry


class BDQNAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        config.lock = mp.Lock()

        self.replay = config.replay_fn()
        self.network = config.network_fn()
        
        if config.tune:
            self.actor = DQNActor(config)
            self.update_actor = True
            filename = config.tune
            state_dict = torch.load('%s.model' % filename, map_location=lambda storage, loc: storage)
            self.network.load_state_dict(state_dict)
            self.activation = {}
            self.network.body.register_forward_hook(self.get_activation('body'))
            self.target_network = self.network
        else:
            self.actor = BDQNActor(config)
            self.update_actor = False
            self.target_network = config.network_fn()
        self.network.share_memory()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_batch_size = 100000
        self.prior_var = config.prior_var
        self.noise_var = config.noise_var
        self.var_k = config.var_k
        self.num_actions = config.action_dim
        self.layer_size = 512
        self.sampled_mean = tensor(torch.normal(0, 0.01, size=(self.num_actions, self.layer_size)))
        self.policy_mean = tensor(torch.normal(0, 0.01, size=(self.num_actions, self.layer_size)))
        self.policy_cov = tensor(torch.normal(0, 1, size=(self.num_actions, self.layer_size, self.layer_size)))
        self.cov_decom = self.policy_cov
        for idx in range(self.num_actions):
            self.policy_cov[idx] = torch.eye(self.layer_size)
            self.cov_decom[idx] = torch.linalg.cholesky((self.policy_cov[idx] + self.policy_cov[idx].T)/2.0)
        self.ppt = tensor(torch.zeros(self.num_actions, self.layer_size, self.layer_size))
        self.py = tensor(torch.zeros(self.num_actions, self.layer_size))
        self.bayes_reg = False
        if not config.tune:
            self.actor.update_mean(self.sampled_mean)
        self.actor.set_network(self.network)
        self.total_steps = 0
        self.num_episodes = 0

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook

    def update_posterior(self):
        self.ppt *= 0
        self.py *= 0
        if self.total_steps > self.config.exploration_steps:
            print('Updating Posterior!')
            v_fac = self.var_k
            self.target_batch_size = min(100000, self.total_steps)
            for sample_idx in range(int(self.target_batch_size/self.config.batch_size)):
                transitions = self.replay.usample()
                states = self.config.state_normalizer(transitions.state)
                next_states = self.config.state_normalizer(transitions.next_state)
                masks = tensor(transitions.mask)
                rewards = tensor(transitions.reward)
                actions = tensor(transitions.action).long()

                with torch.no_grad():
                    policy_state_rep, _, q_target =  self.find_qvals(states, next_states, masks, rewards, actions)
                
                for idx in range(self.config.batch_size):
                    self.ppt[int(actions[idx])] += torch.matmul(policy_state_rep[idx].unsqueeze(0).T, policy_state_rep[idx].unsqueeze(0))
                    self.py[int(actions[idx])] += policy_state_rep[idx].T * q_target[idx]

            for idx in range(self.num_actions):
                inv = torch.inverse(self.ppt[idx]/self.noise_var + 1/self.prior_var*tensor(torch.eye(self.layer_size)))
                self.policy_mean[idx] = torch.matmul(inv, self.py[idx])/self.noise_var
                self.policy_cov[idx] = v_fac * inv
            
#             print(self.policy_mean)
            for idx in range(self.num_actions):
                self.cov_decom[idx] = torch.linalg.cholesky((self.policy_cov[idx]+self.policy_cov[idx].T)/2.0)

    def thompson_sample(self):
        for idx in range(self.num_actions):
            sample = tensor(torch.normal(0, 1, size=(self.layer_size, 1)))
#             self.sampled_mean[idx] = self.policy_mean[idx]
            self.sampled_mean[idx] = self.policy_mean[idx] + torch.matmul(self.cov_decom[idx], sample)[:,0]
        self.actor.update_mean(self.sampled_mean)

    def close(self):
        close_obj(self.replay)
        close_obj(self.actor)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        with torch.no_grad():
            q = self.network(state)['q']
        if self.update_actor == False:
            action = to_np(torch.argmax(torch.matmul(q, self.policy_mean.T), dim=-1))
        else:
            action = to_np(q.argmax(-1))
        self.config.state_normalizer.unset_read_only()
        return action

    def reduce_loss(self, loss):
        return loss.pow(2).mul(0.5).mean()
    
    def find_qvals(self, states, next_states, masks, rewards, actions):
        config = self.config
        with torch.no_grad():
            q_next = torch.matmul(self.target_network(next_states)['q'], self.policy_mean.T)
            if self.config.double_q:
                best_actions = torch.argmax(torch.matmul(self.network(next_states)['q'], self.sampled_mean.T), dim=-1)
                q_next = q_next.gather(1, best_actions.unsqueeze(-1)).squeeze(1)
            else:
                q_next = q_next.max(1)[0]
        policy_state_rep = self.network(states)['q']
        q_target = rewards + self.config.discount ** config.n_step * q_next * masks
        q = torch.matmul(policy_state_rep, self.policy_mean.T)
        q = q.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        return policy_state_rep, q, q_target # missing a term here

    def compute_loss(self, transitions):
        states = self.config.state_normalizer(transitions.state)
        next_states = self.config.state_normalizer(transitions.next_state)
        masks = tensor(transitions.mask)
        rewards = tensor(transitions.reward)
        actions = tensor(transitions.action).long()
        _, q, q_target = self.find_qvals(states, next_states, masks, rewards, actions)
        loss = q_target - q
        return loss

    def step(self):
        config = self.config
        transitions = self.actor.step()
        for states, actions, rewards, next_states, dones, info in transitions:
            self.record_online_return(info)
            if info[0]['episodic_return'] is not None:
                self.num_episodes += 1
                print(self.num_episodes)
                if self.bayes_reg == True:
                    self.update_posterior()
                    self.bayes_reg = False
                if self.update_actor == False:
                    self.thompson_sample()
            self.total_steps += 1
            state_ = np.array([s[-1] if isinstance(s, LazyFrames) else s for s in states])
            self.replay.feed(dict(
                state=state_,
                action=actions,
                reward=[config.reward_normalizer(r) for r in rewards],
                mask=1 - np.asarray(dones, dtype=np.int32),
            ))
#             print(state_)

        if self.total_steps > self.config.exploration_steps and not config.tune:
            transitions = self.replay.sample()
            loss = self.compute_loss(transitions)
            if isinstance(transitions, PrioritizedTransition):
                priorities = loss.abs().add(config.replay_eps).pow(config.replay_alpha)
                idxs = tensor(transitions.idx).long()
                self.replay.update_priorities(zip(to_np(idxs), to_np(priorities)))
                sampling_probs = tensor(transitions.sampling_prob)
                weights = sampling_probs.mul(sampling_probs.size(0)).add(1e-6).pow(-config.replay_beta())
                weights = weights / weights.max()
                loss = loss.mul(weights)

            loss = self.reduce_loss(loss)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
            with config.lock:
                self.optimizer.step()

        if self.total_steps / self.config.sgd_update_frequency \
                % self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        
        if self.total_steps / self.config.sgd_update_frequency \
                % self.config.bdqn_learn_frequency == 0:
            self.bayes_reg = True