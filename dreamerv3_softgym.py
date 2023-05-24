def main():

  import warnings
  import dreamerv3
  from dreamerv3 import embodied
  warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

  # See configs.yaml for all options.
  config = embodied.Config(dreamerv3.configs['defaults'])
  config = config.update(dreamerv3.configs['medium'])
  config = config.update({
      'logdir': './logdir/cloth_flatten/1',
      # 'batch_size': 8,
      # 'batch_length': 16,
      # 'replay_size': 1e4,
      # 'run.actor_batch': 2,
      # 'rssm.deter': 64,
      # '.*\.cnn_depth': 8,
      # '.*\.cnn_depth': 8,
      # '.*\.units': 32,
      # '.*\.layers': 2,
      'jax.policy_devices': [0],
      'jax.train_devices': [0],
      'jax.platform': 'gpu',
      'jax.prealloc': False,
      'run.train_ratio': 64,
      'run.log_every': 300,  # Seconds
      'run.save_every': 60,
      'encoder.mlp_keys': '$^',
      'decoder.mlp_keys': '$^',
      'encoder.cnn_keys': 'image',
      'decoder.cnn_keys': 'image',
      # 'rssm': {'deter': 32, 'stoch': 4, 'classes': 4},      
  })
  config = embodied.Flags(config).parse()

  logdir = embodied.Path(config.logdir)
  step = embodied.Counter()
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(),
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      embodied.logger.TensorBoardOutput(logdir),
      # embodied.logger.WandBOutput(logdir.name, config),
      # embodied.logger.MLFlowOutput(logdir.name),
  ])
    
  from softgym import envs, registered_env
  env_kwargs = registered_env.env_arg_dict['ClothFlatten']
  env_kwargs['observation_mode'] = 'cam_rgb'
  env_kwargs['action_mode'] = 'picker'
  env_kwargs['num_picker'] = 1
  env_kwargs['use_cached_states'] = True
  env_kwargs['save_cached_states'] = True
  env_kwargs['num_variations'] = 10
  env_kwargs['camera_height'] = 128
  env_kwargs['camera_width'] = 128
  # env_kwargs['render'] = False
  # env_kwargs['horizon'] = 10
  
  import os.path as osp
  env = envs.cloth_flatten.ClothFlattenEnv( \
    cached_states_path=osp.join( \
      osp.dirname(osp.abspath(__file__)), \
      'cached_initial_states', \
      'cloth_flatten_init_states.pkl' \
    ), **env_kwargs)
  
  from embodied.envs import from_gym
  env = from_gym.FromGym(env, obs_key='image')  # Or obs_key='vector'.
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
