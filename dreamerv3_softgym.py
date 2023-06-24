import warnings
import dreamerv3
from dreamerv3 import embodied
from dreamerv3.embodied.envs import from_gym
from softgym import registered_env
from my_cloth_flatten_env import MyClothFlattenEnv

def main():
  warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

  # See configs.yaml for all options.
  config = embodied.Config(dreamerv3.configs['defaults'])
  config = config.update(dreamerv3.configs['small'])
  config = config.update({
      'logdir': './logdir/cloth_flatten/10',
      'jax.policy_devices': [0],
      'jax.train_devices': [0],
      'jax.platform': 'gpu',
      'jax.prealloc': False,
      'run.train_ratio': 64,
      'run.log_every': 300,  # Seconds
      'run.save_every': 120,
      'envs.amount': 1,
      'envs.parallel': 'none',
      'encoder.mlp_keys': '$^',
      'decoder.mlp_keys': '$^',
      'encoder.cnn_keys': 'image',
      'decoder.cnn_keys': 'image',
  })
  config = embodied.Flags(config).parse()

  logdir = embodied.Path(config.logdir)
  step = embodied.Counter()
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(),
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      embodied.logger.TensorBoardOutput(logdir)
  ])

  env_kwargs = registered_env.env_arg_dict['ClothFlatten']
  env_kwargs['observation_mode'] = 'cam_rgb'
  env_kwargs['action_mode'] = 'pickerpickplace'
  env_kwargs['action_repeat'] = 1
  env_kwargs['num_picker'] = 1
  env_kwargs['use_cached_states'] = True
  env_kwargs['save_cached_states'] = True
  env_kwargs['num_variations'] = 100
  env_kwargs['camera_height'] = 128
  env_kwargs['camera_width'] = 128
  
  env = MyClothFlattenEnv(**env_kwargs)
  env = from_gym.FromGym(env, obs_key='image')
  env = dreamerv3.wrap_env(env, config)
  env = embodied.BatchEnv([env], parallel=False)
  
  agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
  replay = embodied.replay.Uniform(
      config.batch_length, config.replay_size, logdir / 'replay')
  args = embodied.Config(
      **config.run, logdir=config.logdir,
      batch_steps=config.batch_size * config.batch_length)
  embodied.run.train(agent, env, replay, logger, args)
  # embodied.run.eval_only(agent, env, logger, args)

if __name__ == '__main__':
  main()
