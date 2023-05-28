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

    def compute_reward(self, action=None, obs=None, set_prev_reward=False):
        reward = super().compute_reward(action, obs, set_prev_reward)
        return reward
    
    def _reset(self):
        obs = super()._reset()
        return obs