from softgym.envs.cloth_flatten import ClothFlattenEnv
import os

class MyClothFlattenEnv(ClothFlattenEnv):
    def __init__(self, **kwargs):
        if kwargs.get('use_cached_states', True) or kwargs.get('save_cached_states', True):
            import os.path as osp
            cached_states_path = kwargs.get('cached_states_path',
                                            osp.join('cached_initial_states', 'cloth_flatten_init_states.pkl'))
            if not osp.isabs(cached_states_path):
                cached_states_path = osp.abspath(cached_states_path)
            if not osp.isdir(osp.dirname(cached_states_path)):
                os.makedirs(osp.dirname(cached_states_path))
        
        super().__init__(cached_states_path=cached_states_path, **kwargs)
        
        self.picking_duration = 0
        self.total_steps = 0

    def compute_reward(self, action=None, obs=None, set_prev_reward=False):
        reward = super().compute_reward(action, obs, set_prev_reward)
        for pp in self.get_picked_particle():
            if pp != -1:
                self.picking_duration += 0.2 * (0.1 ** (1 / 10000)) ** self.total_steps
            else:
                self.picking_duration = 0
        self.total_steps += 1
        reward += self.picking_duration
        return reward
    
    def _reset(self):
        self.picking_duration = 0
        return super()._reset()