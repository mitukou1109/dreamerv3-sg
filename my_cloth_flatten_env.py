from softgym.envs.cloth_flatten import ClothFlattenEnv
import os
import pyflex

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
        
        self.prev_covered_area = None

    def compute_reward(self, action=None, obs=None, set_prev_reward=False):
        curr_covered_area = self._get_current_covered_area(pyflex.get_positions())
        reward = curr_covered_area
        if any([pp != -1 for pp in self.get_picked_particle()]):
            reward += 1.5 * (curr_covered_area - self.prev_covered_area)
            # print(curr_covered_area - self.prev_covered_area)
        self.prev_covered_area = curr_covered_area
        return reward
    
    def _reset(self):
        obs = super()._reset()
        self.prev_covered_area = self._get_current_covered_area(pyflex.get_positions())
        return obs