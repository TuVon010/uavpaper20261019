"""Independent vectorized pure-RL screening and six-action hybrid training."""
import copy,signal,time
from dataclasses import asdict,dataclass
from pathlib import Path
import numpy as np
import torch
from .config import Config
from .environment import UAVMECEnv
from .learners import SAC,SACConfig,DeterministicConfig,DeterministicLearner,PPO,PPOConfig,Replay,seed_everything
from .vector import VectorEnvs
from .io import write_json,append_csv,capture_rng,restore_rng,save_checkpoint,load_checkpoint,source_manifest,clean
from .episode_log import EpisodeAccumulator,normalized_delay

@dataclass(frozen=True)
class TrainConfig:
    algorithm:str='sac'
    control:str='pure'
    seed:int=1
    num_envs:int=8
    use_processes:bool=True
    device:str='cuda'
    hidden:int=128
    batch_size:int=256
    replay_capacity:int=100000
    random_steps:int=512
    critic_warmup_steps:int=512
    actor_lr:float=1e-4
    critic_lr:float=3e-4
    gamma:float=1.
    tau:float=.005
    alpha_lr:float=1e-4
    initial_alpha:float=.01
    exploration_sigma:float=.1
    ppo_rollout_frames:int=400
    ppo_epochs:int=10
    ppo_critic_epochs:int=10
    ppo_minibatch_size:int=128
    ppo_lr:float=3e-4
    ppo_critic_lr:float|None=None
    ppo_gae_lambda:float=.95
    ppo_clip_ratio:float=.2
    ppo_value_clip:float|None=None
    ppo_target_kl:float|None=.02
    ppo_entropy_coef:float=0.
    ppo_value_coef:float=.5
    ppo_max_grad_norm:float=.5
    ppo_motion_memory:bool=False
    ppo_known_delay_feature:bool=False
    ppo_ar_rho:float=0.
    ppo_continuation_critic:bool=False
    ppo_continuation_mode:str='known'
    ppo_ar_adaptive:bool=False
    ppo_ar_rho_max:float=.9
    ppo_schedule_steps:int=0
    ppo_noise_final_scale:float=1.
    ppo_noise_anneal_start:float=.1
    ppo_noise_anneal_end:float=.7
    ppo_actor_lr_final_scale:float=1.
    selection_mode:str='legacy'
    ppo_variant:str='baseline'
    ppo_popart_beta:float=.01
    ppo_popart_min_std:float=.1
    eval_every:int=3000
    eval_episodes:int=3
    eval_start:int=20000
    checkpoint_every:int=3000
    log_frames:bool=False
    render_every_episodes:int=10

    def __post_init__(self):
        positive_integers=('num_envs','hidden','batch_size','replay_capacity','ppo_rollout_frames',
            'ppo_epochs','ppo_critic_epochs','ppo_minibatch_size','eval_every','eval_episodes','checkpoint_every')
        for name in positive_integers:
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or value<1:
                raise ValueError(name+' must be a positive integer')
        for name in ('seed','random_steps','critic_warmup_steps','eval_start','ppo_schedule_steps'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or value<0:
                raise ValueError(name+' must be a nonnegative integer')
        if not all(np.isfinite(value) for value in
                   (self.ppo_clip_ratio,self.ppo_entropy_coef,self.ppo_value_coef,self.ppo_max_grad_norm)):
            raise ValueError('PPO loss and gradient settings must be finite')
        if not 0<self.ppo_clip_ratio<1 or self.ppo_entropy_coef<0 or min(self.ppo_value_coef,self.ppo_max_grad_norm)<=0:
            raise ValueError('Invalid PPO clipping, loss coefficient or gradient bound')
        for name in ('ppo_value_clip','ppo_target_kl'):
            value=getattr(self,name)
            if value is not None and (not np.isfinite(value) or value<=0):
                raise ValueError(name+' must be positive and finite or None')
        if self.algorithm not in {'sac','td3','ddpg','ppo'}:raise ValueError('Unknown algorithm')
        if self.control not in {'pure','hybrid'}:raise ValueError('Unknown control interface')
        if self.ppo_variant not in {'baseline','value_decomp','value_decomp_popart'}:raise ValueError('Unknown PPO variant')
        if self.algorithm!='ppo' and self.ppo_variant!='baseline':raise ValueError('Architecture variants require PPO')
        if min(self.num_envs,self.eval_every,self.checkpoint_every,self.ppo_rollout_frames)<1:raise ValueError('Periods must be positive')
        if any(x%self.num_envs for x in (self.eval_every,self.checkpoint_every,self.ppo_rollout_frames)):
            raise ValueError('Periods must be divisible by vector count')
        if self.ppo_known_delay_feature and (self.algorithm!='ppo' or self.control!='hybrid'):
            raise ValueError('Known service cost is available only to hybrid PPO')
        if self.ppo_motion_memory and self.algorithm!='ppo':
            raise ValueError('Explicit motion memory requires PPO')
        if self.ppo_ar_rho and not self.ppo_motion_memory:
            raise ValueError('AR motion needs explicit previous-request memory')
        if self.ppo_continuation_mode not in {'known','learned'}:
            raise ValueError('Unknown continuation mode')
        if self.ppo_continuation_critic:
            if self.ppo_variant!='value_decomp_popart':
                raise ValueError('Continuation critic requires PopArt')
            if self.ppo_continuation_mode=='known' and not self.ppo_known_delay_feature:
                raise ValueError('Known continuation critic needs known service cost')
            if self.ppo_continuation_mode=='learned' and self.ppo_known_delay_feature:
                raise ValueError('Learned continuation critic must predict the current service cost')
        if self.selection_mode not in {'legacy','feasible_return'}:
            raise ValueError('Unknown checkpoint selection protocol')
        if (self.ppo_ar_adaptive or self.ppo_schedule_steps) and self.algorithm!='ppo':
            raise ValueError('AR/schedule modifications require PPO')
        if self.ppo_ar_adaptive and not self.ppo_motion_memory:
            raise ValueError('Adaptive AR requires observable motion memory')
        if not isinstance(self.render_every_episodes,int) or self.render_every_episodes<0:
            raise ValueError('Render period must be a nonnegative episode count')

def learner_for(obs_size,action_size,run,physics=None):
    if run.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA requested but not available')
    if run.algorithm=='sac':
        return SAC(obs_size,action_size,SACConfig(hidden=run.hidden,actor_lr=run.actor_lr,
            critic_lr=run.critic_lr,alpha_lr=run.alpha_lr,initial_alpha=run.initial_alpha,
            gamma=run.gamma,tau=run.tau,batch_size=run.batch_size,replay_capacity=run.replay_capacity),device=run.device)
    if run.algorithm=='ppo':
        return PPO(obs_size,action_size,PPOConfig(hidden=run.hidden,actor_lr=run.ppo_lr,
            critic_lr=run.ppo_lr if run.ppo_critic_lr is None else run.ppo_critic_lr,
            gamma=run.gamma,gae_lambda=run.ppo_gae_lambda,rollout_steps=run.ppo_rollout_frames,
            clip_ratio=run.ppo_clip_ratio,value_clip=run.ppo_value_clip,target_kl=run.ppo_target_kl,
            entropy_coef=run.ppo_entropy_coef,value_coef=run.ppo_value_coef,max_grad_norm=run.ppo_max_grad_norm,
            motion_memory=run.ppo_motion_memory,ar_rho=run.ppo_ar_rho,
            known_delay_index=obs_size-1 if run.ppo_known_delay_feature else None,
            continuation_critic=run.ppo_continuation_critic,
            continuation_mode=run.ppo_continuation_mode,
            ar_adaptive=run.ppo_ar_adaptive,ar_rho_max=run.ppo_ar_rho_max,
            schedule_steps=run.ppo_schedule_steps,noise_final_scale=run.ppo_noise_final_scale,
            noise_anneal_start=run.ppo_noise_anneal_start,noise_anneal_end=run.ppo_noise_anneal_end,
            actor_lr_final_scale=run.ppo_actor_lr_final_scale,
            epochs=run.ppo_epochs,minibatch_size=run.ppo_minibatch_size,
            critic_epochs=run.ppo_critic_epochs,popart_beta=run.ppo_popart_beta,popart_min_std=run.ppo_popart_min_std,
            variant=run.ppo_variant),
            device=run.device,physics=physics)
    return DeterministicLearner(obs_size,action_size,DeterministicConfig(algorithm=run.algorithm,
        hidden=run.hidden,actor_lr=run.actor_lr,critic_lr=run.critic_lr,gamma=run.gamma,
        tau=run.tau,batch_size=run.batch_size,replay_capacity=run.replay_capacity),device=run.device)

def _cost(info):
    return normalized_delay(info)


def augment_observations(base,previous_raw,known_delays,run):
    """Explicit causal policy context; never stores a hidden noise/history state."""
    result=np.asarray(base,dtype=np.float32)
    single=result.ndim==1
    if single:result=result[None,:]
    if result.ndim!=2:raise ValueError('Observation batch must be a matrix')
    parts=[result]
    if run.ppo_motion_memory:
        memory=np.asarray(previous_raw,dtype=np.float32).reshape(len(result),-1)
        if memory.shape[1]<6:raise ValueError('Six previous velocity latents required')
        parts.append(memory[:,:6])
    if run.ppo_known_delay_feature:
        parts.append(np.asarray(known_delays,dtype=np.float32).reshape(len(result),1))
    result=np.concatenate(parts,axis=1)
    if not np.isfinite(result).all():raise ValueError('Non-finite policy observation')
    return result[0] if single else result

def evaluate(cfg,learner=None,control='pure',episodes=5,episode_start=30000,seed=1,out=None,baseline=None,
             deterministic=True,policy_seed=1):
    env=UAVMECEnv(cfg,seed=seed,control=control if baseline is None else baseline)
    if out is not None:
        out=Path(out)
        if any((out/name).exists() for name in ('summary.json','trajectories.jsonl','episodes.csv')):
            raise FileExistsError('Evaluation exists or is incomplete; preserve it and use a new directory')
        out.mkdir(parents=True,exist_ok=True)
    from types import SimpleNamespace
    learner_cfg=getattr(learner,'cfg',None)
    context=SimpleNamespace(ppo_motion_memory=bool(getattr(learner_cfg,'motion_memory',False)),
        ppo_known_delay_feature=getattr(learner_cfg,'known_delay_index',None) is not None)
    generator=None
    if learner is not None and not deterministic:
        generator=torch.Generator(device=learner.device).manual_seed(policy_seed)
    records=[];scenarios=[]
    for episode in range(episode_start,episode_start+episodes):
        obs=env.reset(episode=episode);rows=[];hashes=[];rewards=[];done=False
        previous_raw=np.zeros(6,np.float32)
        while not done:
            known_delay=env.prepare_service() if context.ppo_known_delay_feature else None
            policy_obs=augment_observations(obs,previous_raw,known_delay,context)
            start=time.perf_counter()
            if learner is None:
                action=np.zeros(env.action_size,np.float32)
            elif context.ppo_motion_memory:
                actions,raws=learner.act_batch(policy_obs[None,:],deterministic=deterministic,
                                              generator=generator,return_raw=True)
                action=actions[0];previous_raw=raws[0,:6].copy()
            elif deterministic:
                action=learner.act(policy_obs,deterministic=True)
            else:
                action=learner.act(policy_obs,deterministic=False,generator=generator)
            policy_wall=time.perf_counter()-start
            obs,reward,terminated,truncated,info=env.step(action);done=terminated or truncated
            row=clean(dict(info));row.update(episode=episode,reward=float(reward),policy_wall_s=policy_wall)
            row['total_decision_wall_s']=policy_wall+float(info.get('decision_wall_s',0.))
            rows.append(row);rewards.append(float(reward));hashes.append(info['frame_scenario_hash'])
            if out is not None:
                import json
                trajectory=dict(row,action=np.asarray(action),user_xy_m=env.user_position,uav_xy_m=env.uav_position)
                with (out/'trajectories.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(clean(trajectory),allow_nan=False)+'\n')
        cumulative=np.asarray(rows[-1]['cumulative_energy_j'],float)
        deadline_ok=all(r['deadline_completion']>=1.-1e-12 and r.get('allocation_invalid_rate',0.)==0 for r in rows)
        budget_ok=bool(np.all(cumulative<=cfg.energy_budget_j+1e-6))
        record={'episode':episode,'scenario_hash':rows[0]['scenario_hash'],'frames':len(rows),
            'weighted_delay_mean':float(np.mean([_cost(r) for r in rows])),
            'weighted_delay_sum_per_frame':20*float(np.mean([_cost(r) for r in rows])),
            'mean_delay_s':float(np.mean([r['mean_delay_s'] for r in rows])),
            'deadline_completion':float(np.mean([r['deadline_completion'] for r in rows])),
            'episode_return':sum(rewards),'per_uav_energy_j':cumulative.tolist(),
            'total_energy_j':float(cumulative.sum()),'budget_feasible':budget_ok,
            'deadline_feasible':deadline_ok,'fully_feasible':budget_ok and deadline_ok,
            'collision_attempt_frames':sum(bool(r['collision_attempt']) for r in rows),
            'boundary_attempt_frames':sum(bool(r['boundary_attempt']) for r in rows),
            'request_safe':all(not r['collision_attempt'] and not r['boundary_attempt'] for r in rows),
            'projection_failure_frames':sum(bool(r.get('projection_failed',False)) for r in rows),
            'deadline_violation_rate':float(np.mean([1-r['deadline_completion'] for r in rows])),
            'deadline_violation_cost':float(np.mean([r['deadline_violation_cost'] for r in rows])),
            'allocation_invalid_rate':float(np.mean([r.get('allocation_invalid_rate',0.) for r in rows])),
            'solver_feasible_frame_fraction':float(np.mean([r.get('solver_feasible',True) for r in rows])),
            'max_energy_overrun_fraction':float(np.maximum(cumulative/cfg.energy_budget_j-1,0).max()),
            'energy_overrun_fraction':float(np.maximum(cumulative/cfg.energy_budget_j-1,0).mean()),
            'total_propulsion_energy_j':float(np.sum(rows[-1]['cumulative_propulsion_energy_j'])),
            'total_compute_energy_j':float(np.sum(rows[-1]['cumulative_compute_energy_j'])),
            'fallback_rate':float(np.mean([r['fallback_rate'] for r in rows])),
            'total_decision_mean_s':float(np.mean([r['total_decision_wall_s'] for r in rows])),
            'total_decision_p95_s':float(np.percentile([r['total_decision_wall_s'] for r in rows],95))}
        records.append(record);scenarios.append({'episode':episode,'scenario_hash':record['scenario_hash'],'frame_scenario_hashes':hashes})
        if out is not None:append_csv(out/'episodes.csv',record)
    summary={key:float(np.mean([r[key] for r in records])) for key in records[0]
        if isinstance(records[0][key],(float,int,bool))}
    summary.update(episodes=episodes,episode_start=episode_start,seed=seed,control=control,baseline=baseline,
        feasible_episodes=sum(r['fully_feasible'] for r in records),
        energy_feasible_episodes=sum(r['budget_feasible'] for r in records),
        deadline_feasible_episodes=sum(r['deadline_feasible'] for r in records),
        environment_config=cfg.to_dict(),source=source_manifest(),
        action_protocol=('deterministic conditional actor mean' if deterministic else 'sampled conditional policy; independent RNG continuous across episodes')+'; pure association uses unchanged keyed categorical routing',
        deterministic_policy=deterministic,policy_seed=policy_seed if not deterministic else None,
        motion_memory=context.ppo_motion_memory,known_delay_feature=context.ppo_known_delay_feature,
        energy_semantics='J per episode; aggregate totals are means of episode totals')
    result={'summary':summary,'episodes':records,'scenarios':scenarios}
    if learner is not None and hasattr(learner,'policy_diagnostics'):
        summary['policy_execution_parameters']=learner.policy_diagnostics()
    if out is not None:write_json(out/'summary.json',result)
    return result

def selection(result,mode='legacy'):
    s=result['summary'];feasible=s['feasible_episodes']==s['episodes']
    if mode=='feasible_return':
        return (0 if feasible else 1,-s['episode_return'],s['weighted_delay_mean'])
    if mode!='legacy':raise ValueError('Unknown selection mode')
    return (0 if feasible else 1,s['weighted_delay_mean'] if feasible else -s['episode_return'])

PENDING_FIELDS=('observation','action','raw_action','log_prob','value','reward','terminated','next_value','stream_id')

def _trim_history(out,step):
    """An interrupted process may have written beyond its last atomic checkpoint."""
    import csv,json
    out=Path(out)
    for name,key in (('train.csv','step'),('validation.csv','step'),('episodes.csv','end_step')):
        path=out/name
        if not path.exists():continue
        with path.open(newline='',encoding='utf-8') as stream:
            reader=csv.DictReader(stream);fields=reader.fieldnames
            rows=[]
            for row in reader:
                # An interrupted append may leave a final incomplete CSV row.
                try:keep=int(row[key])<=step and None not in row and all(v is not None for v in row.values())
                except (ValueError,TypeError):continue
                if keep:rows.append(row)
        with path.open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    path=out/'updates.jsonl'
    if path.exists():
        lines=path.read_text().splitlines();rows=[]
        for index,row in enumerate(lines):
            try:value=json.loads(row)
            except json.JSONDecodeError:
                if index==len(lines)-1:break
                raise
            if int(value['step'])<=step:rows.append(row)
        path.write_text(''.join(r+'\n' for r in rows),encoding='utf-8')
    for path in (out/'validation').glob('step_*.json'):
        if int(path.stem.removeprefix('step_'))>step:path.unlink()

def train(cfg,out,steps,run=None,resume=None):
    run=run or TrainConfig();cfg.validate()
    if steps%run.num_envs or steps<1:raise ValueError('Target steps must be positive and divisible by num_envs')
    torch.set_num_threads(1);seed_everything(run.seed)
    out=Path(out);out.mkdir(parents=True,exist_ok=True);source=source_manifest()
    pool=VectorEnvs(cfg,run.seed,run.control,run.num_envs,run.use_processes)
    action_rng=np.random.default_rng(run.seed+202)
    initial_base=np.asarray(pool.all('reset',list(range(run.num_envs))))
    initial_cost=pool.all('prepare',[None]*run.num_envs) if run.ppo_known_delay_feature else None
    observations=augment_observations(initial_base,np.zeros((run.num_envs,6),np.float32),initial_cost,run)
    observation_size=observations.shape[1]
    learner=learner_for(observation_size,pool.action_size,run,cfg.to_dict())
    replay=None if run.algorithm=='ppo' else Replay(observation_size,pool.action_size,run.replay_capacity,run.seed+101)
    episode_ids=list(range(run.num_envs));next_episode=run.num_envs
    returns=np.zeros(run.num_envs);lengths=np.zeros(run.num_envs,int)
    episode_accumulators=[EpisodeAccumulator(episode_ids[i],i) for i in range(run.num_envs)]
    completed_episodes=0
    pending_fields=PENDING_FIELDS
    decomposed=run.algorithm=='ppo' and run.ppo_variant in {'value_decomp','value_decomp_popart'}
    if decomposed:pending_fields+=('reward_components','value_components','next_value_components')
    pending={k:[] for k in pending_fields};step=0;best_key=None;best_step=None
    try:
        if resume is not None:
            cp=load_checkpoint(resume)
            if cp.get('evaluation_only'):raise ValueError('Resume requires last.pt, not an evaluation checkpoint')
            if cp['version']!=1 or cp['source']!=source:raise ValueError('Checkpoint source/version differs')
            if cp['train_config']!=asdict(run) or cp['config']!=cfg.to_dict():raise ValueError('Resume configuration differs')
            if cp['step']>steps:raise ValueError('Target below current step')
            learner.load_state_dict(cp['learner'])
            if replay is not None:replay.load_state_dict(cp['replay'])
            pool.all('load',cp['environments']);observations=cp['observations'].copy()
            action_rng.bit_generator.state=copy.deepcopy(cp['action_rng'])
            episode_ids=cp['episode_ids'];next_episode=cp['next_episode'];returns=cp['returns'];lengths=cp['lengths']
            if 'episode_accumulators' not in cp:
                raise ValueError('Resume checkpoint lacks exact episode logging state')
            episode_accumulators=[EpisodeAccumulator.from_state_dict(s) for s in cp['episode_accumulators']]
            completed_episodes=cp['completed_episodes']
            pending=cp['pending'];step=cp['step'];best_key=cp['best_key'];best_step=cp['best_step']
            origin=Path(resume).resolve().parent
            if best_key is not None:
                selected_path=origin/'best_checkpoints'/('step_%08d.pt'%best_step)
                if not selected_path.exists():selected_path=origin/'best.pt'
                selected=load_checkpoint(selected_path)
                if selected['step']!=best_step or selected['step']>step or tuple(selected['best_key'])!=tuple(best_key):raise ValueError('Inconsistent best checkpoint')
            if origin!=out.resolve():
                if any((out/name).exists() for name in ('train.csv','episodes.csv','updates.jsonl','validation.csv','last.pt')):raise FileExistsError('Resume output exists')
                import shutil
                for name in ('train.csv','episodes.csv','updates.jsonl','validation.csv','best.pt'):
                    if (origin/name).exists():shutil.copy2(origin/name,out/name)
                if (origin/'validation').exists():shutil.copytree(origin/'validation',out/'validation',dirs_exist_ok=True)
                if (origin/'best_checkpoints').exists():shutil.copytree(origin/'best_checkpoints',out/'best_checkpoints',dirs_exist_ok=True)
            if best_key is not None:
                import shutil
                if selected_path.resolve()!=(out/'best.pt').resolve():shutil.copy2(selected_path,out/'best.pt')
            _trim_history(out,step)
            restore_rng(cp['rng'])
        elif any((out/name).exists() for name in ('last.pt','train.csv','episodes.csv','updates.jsonl','validation.csv')):raise FileExistsError('Run exists')
        manifest={'source':source,'config':cfg.to_dict(),'train_config':asdict(run),
            'learner_config':asdict(learner.cfg),'observation_size':observation_size,'base_observation_size':pool.observation_size,'action_size':pool.action_size,
            'steps_target':steps,'resume':str(resume) if resume else None,'torch':torch.__version__,
            'cuda_build':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if run.device.startswith('cuda') else None,
            'physical_seed':run.seed,'pure_resource_optimizer_used':False if run.control=='pure' else True,
            'value_semantics':{'sac':'entropy-regularized softQ','ppo':'GAE target fit, not independent MC value accuracy',
                'td3':'unregularized deterministic targetQ','ddpg':'unregularized deterministic targetQ'}[run.algorithm],
            'ppo_variant':run.ppo_variant,
            'parameters':{'actor':sum(p.numel() for p in learner.actor.parameters()),
                'critic':sum(p.numel() for p in learner.critic.parameters()) if hasattr(learner,'critic') else None},
            'reward_changed':False,'resource_solver_changed':False}
        manifest['selection_mode']=run.selection_mode
        manifest['policy_context']={'previous_motion_latent':run.ppo_motion_memory,'current_solver_cost':run.ppo_known_delay_feature}
        manifest['schedule_semantics']='Fixed run-config horizon; progress changes only before an empty PPO rollout; active scales saved in learner checkpoints.'
        manifest['training_log_semantics']={'episodes.csv':'One completed training episode per row; end_step is the last global transition index.',
            'train.csv':'Optional frame diagnostics; enabled only by log_frames.',
            'decision_time':'Policy batch wall time divided by num_envs plus each environment decision wall time; excludes learner updates and plotting.'}
        write_json(out/'manifest.json',manifest)
        initial_step=step;started=time.perf_counter();rollout_wall=update_wall=validation_wall=plot_wall=0.;stop=[None];handlers={}
        last_rendered_episode=completed_episodes;plot_error=None
        def on_signal(signum,_):stop[0]=int(signum)
        try:
            for sig in (signal.SIGTERM,signal.SIGINT):handlers[sig]=signal.signal(sig,on_signal)
        except ValueError:pass
        def checkpoint(path,evaluation_only=False):
            state=dict(version=1,source=source,config=cfg.to_dict(),train_config=asdict(run),step=step,
                learner=learner.state_dict(),best_key=best_key,best_step=best_step,
                observation_size=observation_size,action_size=pool.action_size)
            if evaluation_only:state['evaluation_only']=True
            else:state.update(replay=None if replay is None else replay.state_dict(),
                environments=pool.all('state',[None]*run.num_envs),observations=observations.copy(),
                action_rng=copy.deepcopy(action_rng.bit_generator.state),rng=capture_rng(),
                episode_ids=copy.deepcopy(episode_ids),next_episode=next_episode,returns=returns.copy(),lengths=lengths.copy(),
                episode_accumulators=[a.state_dict() for a in episode_accumulators],completed_episodes=completed_episodes,
                pending=copy.deepcopy(pending))
            save_checkpoint(path,state)
        def render_curves():
            nonlocal plot_wall,plot_error,last_rendered_episode
            clock=time.perf_counter();rng=capture_rng()
            try:
                from . import reporting
                renderer=getattr(reporting,'render_run_curves',None)
                if renderer is not None:renderer(out)
                plot_error=None;last_rendered_episode=completed_episodes
            except ImportError as error:
                plot_error=type(error).__name__+': '+str(error)
            except Exception as error:
                # Plotting failures must not discard a costly training session.
                import warnings
                plot_error=type(error).__name__+': '+str(error)
                warnings.warn('Training curves could not be rendered: '+plot_error,RuntimeWarning)
            finally:
                restore_rng(rng);plot_wall+=time.perf_counter()-clock
        def status(state,error=None):
            elapsed=time.perf_counter()-started
            value=dict(state=state,step=step,target_steps=steps,session_steps=step-initial_step,
                session_wall_s=elapsed,session_fps=(step-initial_step)/max(elapsed,1e-12),
                rollout_wall_s=rollout_wall,update_wall_s=update_wall,validation_wall_s=validation_wall,
                plot_wall_s=plot_wall,plot_error=plot_error,completed_episodes=completed_episodes,
                rollout_fps=(step-initial_step)/max(rollout_wall,1e-12),learner_updates=learner.updates,
                pending_rollout=len(pending['reward']),best_key=best_key,best_step=best_step,
                training_complete=state=='training_complete',evaluation_complete=False,error=error,signal=stop[0])
            if run.algorithm=='ppo' and hasattr(learner,'policy_diagnostics'):
                value['policy_execution_parameters']=learner.policy_diagnostics()
            write_json(out/'status.json',value);return value
        status('running')
        try:
            while step<steps and stop[0] is None:
                clock=time.perf_counter()
                if run.algorithm=='ppo' and run.ppo_schedule_steps>0 and not pending['reward']:
                    # Every on-policy rollout has one declared conditional policy.
                    # A resumed partial rollout keeps its checkpointed scale.
                    learner.set_training_progress(step)
                policy_started=time.perf_counter()
                if run.algorithm=='ppo':
                    if decomposed:
                        actions,logp,values,raw,component_values=learner.act_train_batch(observations,return_components=True)
                    else:actions,logp,values,raw=learner.act_train_batch(observations)
                elif step<run.random_steps:actions=action_rng.uniform(-1,1,(run.num_envs,pool.action_size)).astype(np.float32)
                else:
                    actions=learner.act_batch(observations,deterministic=run.algorithm!='sac')
                    if run.algorithm in {'ddpg','td3'}:
                        actions=np.clip(actions+action_rng.normal(0,run.exploration_sigma,actions.shape),-1,1).astype(np.float32)
                policy_wall_s=(time.perf_counter()-policy_started)/run.num_envs
                results=pool.all('step',actions)
                next_obs=np.asarray([r[0] for r in results]);reward=np.asarray([r[1] for r in results]);terminal=np.asarray([r[2] or r[3] for r in results])
                next_cost=pool.all('prepare',[None]*run.num_envs) if run.ppo_known_delay_feature else None
                next_obs=augment_observations(next_obs,raw[:,:6] if run.ppo_motion_memory else None,next_cost,run)
                if replay is not None:replay.add_batch(observations,actions,reward,next_obs,terminal)
                else:
                    if decomposed:
                        from .rl.value_components import reward_components
                        component_rewards=reward_components([r[4] for r in results],reward)
                        next_components=learner.value_components_batch(next_obs)
                        next_components=np.where(terminal[:,None],0.,next_components)
                        next_values=next_components.sum(-1)
                    else:
                        with torch.no_grad():next_values=learner.critic(torch.as_tensor(next_obs,dtype=torch.float32,device=learner.device)).reshape(-1).cpu().numpy()
                    next_values=np.where(terminal,0.,next_values)
                    for i in range(run.num_envs):
                        items=(observations[i].copy(),actions[i].copy(),raw[i].copy(),float(logp[i]),float(values[i]),float(reward[i]),bool(terminal[i]),float(next_values[i]),i)
                        if decomposed:items+=(component_rewards[i].copy(),component_values[i].copy(),next_components[i].copy())
                        for k,value in zip(pending_fields,items):pending[k].append(value)
                returns+=reward;lengths+=1
                for i,result in enumerate(results):
                    info=result[4]
                    episode_accumulators[i].add(reward[i],info,step+i+1,policy_wall_s)
                    if run.log_frames:append_csv(out/'train.csv',dict(step=step+i+1,episode=episode_ids[i],frame=int(lengths[i]-1),
                        reward=float(reward[i]),episode_return_so_far=float(returns[i]),terminated=bool(terminal[i]),
                        scenario_hash=info['scenario_hash'],frame_scenario_hash=info['frame_scenario_hash'],
                        weighted_delay_mean=_cost(info),mean_delay_s=info['mean_delay_s'],deadline_completion=info['deadline_completion'],
                        deadline_violation_cost=info['deadline_violation_cost'],allocation_invalid_rate=info.get('allocation_invalid_rate',0.),
                        solver_feasible=info.get('solver_feasible',True),solver_wall_s=info['solver_wall_s'],
                        energy_j=info['total_energy_j'],per_uav_energy_j=info['energy_j'],propulsion_energy_j=info['propulsion_energy_j'],compute_energy_j=info['compute_energy_j'],
                        cumulative_energy_j=info['cumulative_energy_j'],association_counts=info.get('association_counts',[]),
                        collision_attempt=info['collision_attempt'],boundary_attempt=info['boundary_attempt'],
                        projection_failed=info.get('projection_failed',False),
                        projection_correction_mps=float(np.mean(info.get('projection_correction_norm_mps',[0.]))),
                        requested_velocity=info['requested_velocity_mps'],executed_velocity=info['uav_velocity'],
                        action_saturation=float(np.mean(np.abs(actions[i])>.98))))
                    if terminal[i]:
                        append_csv(out/'episodes.csv',episode_accumulators[i].record(step+i+1,cfg.energy_budget_j))
                        completed_episodes+=1
                        reset_base=pool.reset_one(i,next_episode)
                        reset_cost=pool.prepare_one(i) if run.ppo_known_delay_feature else None
                        next_obs[i]=augment_observations(reset_base,np.zeros(6,np.float32),reset_cost,run)
                        episode_ids[i]=next_episode;next_episode+=1;returns[i]=0.;lengths[i]=0
                        episode_accumulators[i]=EpisodeAccumulator(episode_ids[i],i)
                observations=next_obs;step+=run.num_envs;rollout_wall+=time.perf_counter()-clock
                metrics=[];clock=time.perf_counter()
                if replay is not None and step>=run.random_steps and replay.size>=run.batch_size:
                    for _ in range(run.num_envs):metrics.append(learner.update(replay,update_actor=step>run.random_steps+run.critic_warmup_steps))
                elif run.algorithm=='ppo' and len(pending['reward'])>=run.ppo_rollout_frames:
                    metrics.append(learner.update({k:np.asarray(v) for k,v in pending.items()}));pending={k:[] for k in pending_fields}
                update_wall+=time.perf_counter()-clock
                if metrics:
                    allkeys=sorted(set().union(*(m.keys() for m in metrics)))
                    aggregate={k:float(np.mean([m[k] for m in metrics if k in m])) for k in allkeys}
                    # JSONL allows algorithm-native diagnostics without sparse CSV columns.
                    import json
                    with (out/'updates.jsonl').open('a') as stream:stream.write(json.dumps(dict(step=step,**aggregate),allow_nan=False)+'\n')
                if step%run.eval_every==0 or (run.selection_mode=='feasible_return' and step==steps and not pending['reward']):
                    clock=time.perf_counter();result=evaluate(cfg,learner,run.control,run.eval_episodes,run.eval_start,run.seed)
                    validation_wall+=time.perf_counter()-clock;key=selection(result,run.selection_mode);improved=best_key is None or key<tuple(best_key)
                    if improved:
                        best_key=key;best_step=step
                        destination=out/'best_checkpoints'/('step_%08d.pt'%step);destination.parent.mkdir(exist_ok=True)
                        checkpoint(destination,evaluation_only=True)
                        import shutil
                        shutil.copy2(destination,out/'best.pt')
                    write_json(out/'validation'/('step_%08d.json'%step),result)
                    s=result['summary'];append_csv(out/'validation.csv',dict(step=step,weighted_delay_mean=s['weighted_delay_mean'],
                        mean_delay_s=s['mean_delay_s'],deadline_completion=s['deadline_completion'],total_energy_j=s['total_energy_j'],
                        feasible_episodes=s['feasible_episodes'],episode_return=s['episode_return'],best_selected=improved))
                    print('%s %s step=%d cost=%.5f feasible=%d/%d fps=%.1f'%(run.algorithm,run.control,step,s['weighted_delay_mean'],s['feasible_episodes'],run.eval_episodes,(step-initial_step)/(time.perf_counter()-started)),flush=True)
                if step%run.checkpoint_every==0 or step%run.eval_every==0:checkpoint(out/'last.pt');status('running')
                if (run.render_every_episodes and
                        completed_episodes//run.render_every_episodes>last_rendered_episode//run.render_every_episodes):
                    render_curves();status('running')
            checkpoint(out/'last.pt')
            if run.render_every_episodes:render_curves()
            return status('paused_signal' if stop[0] is not None else 'training_complete')
        except BaseException as error:status('failed',type(error).__name__+': '+str(error));raise
        finally:
            for sig,handler in handlers.items():signal.signal(sig,handler)
    finally:pool.close()

def load_learner(path,device=None):
    cp=load_checkpoint(path)
    if cp['source']!=source_manifest():raise ValueError('Checkpoint source differs')
    run=TrainConfig(**cp['train_config'])
    if device is not None:
        from dataclasses import replace
        run=replace(run,device=device)
    learner=learner_for(cp['observation_size'],cp['action_size'],run,cp['config']);learner.load_state_dict(cp['learner'])
    return learner,Config(**cp['config']),cp
