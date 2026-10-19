import numpy as np
import pytest
from priority_mec.config import Config
from priority_mec.environment import UAVMECEnv
from priority_mec.scenarios import Scenario, gauss_markov_update
from priority_mec.physics import (initial_uav_positions, minimum_separation, safe_velocity,
                                  propulsion_power, satellite_geometry, channel_parameters)


def test_signed_gauss_markov_and_unwrapped_angle():
    c=Config()
    speed,angle=gauss_markov_update(np.array([0.]),np.array([8.]),np.array([0.]),np.array([1.]),np.array([-4.]),np.array([2.]),c)
    assert speed[0] == pytest.approx(-3.2)
    assert angle[0] == pytest.approx(.9*8+.1+np.sqrt(1-.9**2)*2)
    assert angle[0]>2*np.pi
    assert speed[0]*np.cos(0)<0


def test_keyed_scenes_and_units():
    c=Config()
    a,b,d=Scenario(c,1,42),Scenario(c,1,42),Scenario(c,1,43)
    assert a.hash==b.hash and a.hash!=d.hash
    assert np.all((a.mean_speed>=0)&(a.mean_speed<1))
    assert a.frame_hash(0)!=a.frame_hash(1)
    assert c.noise_dbm_hz==pytest.approx(-174.)
    assert np.std(a.angle_noise)==pytest.approx(2,rel=.08)
    assert np.var(a.speed_noise)==pytest.approx(2,rel=.1)


def test_geometry_and_monotone_power():
    c=Config()
    p=initial_uav_positions(c)
    np.testing.assert_allclose(np.linalg.norm(p-500,axis=1),500)
    assert p[0,0]==1000
    power=propulsion_power(np.linspace(0,15,100),c)
    assert np.all(np.diff(power)>0)
    assert power[0]*60==pytest.approx(10109.4)
    distance,el=satellite_geometry(np.array([[500.,500.]]),30,c)
    assert distance[0]==pytest.approx(600e3)
    assert el[0]==pytest.approx(90)


def test_continuous_safety_and_boundary_prefix():
    c=Config()
    p=np.array([[450.,500.],[520.,500.],[800.,800.]])
    v=np.array([[15.,0.],[-15.,0.],[0.,0.]])
    executed,info=safe_velocity(p,v,c)
    assert info['collision_attempt']
    assert 0<info['safety_scale']<1
    assert minimum_separation(p,executed,1)>=50-1e-10
    start=initial_uav_positions(c)
    executed,info=safe_velocity(start,np.ones((3,2))*15,c)
    assert info['boundary_attempt'] and info['safety_scale']==1
    assert executed[0,0]==0
    assert np.linalg.norm(executed[1])>0 and np.linalg.norm(executed[2])>0
    assert np.all(start+executed>=0) and np.all(start+executed<=1000)


def test_boundary_projection_does_not_freeze_other_uavs():
    c=Config()
    position=initial_uav_positions(c)
    request=np.array([[15.,0.],[5.,-5.],[5.,5.]])
    actual,info=safe_velocity(position,request,c)
    np.testing.assert_array_equal(actual[0],[0.,0.])
    np.testing.assert_array_equal(actual[1:],request[1:])
    assert info['boundary_attempt'] and not info['collision_attempt']
    assert info['boundary_projected_uavs']==[True,False,False]
    assert info['safety_scale']==1
    assert minimum_separation(position,actual,c.frame_s)>=c.safe_distance_m
    assert np.all(np.linalg.norm(actual,axis=1)<=c.speed_max_mps)


def test_pure_never_calls_solver_or_hides_deadline_violation(monkeypatch):
    import priority_mec.solvers as solvers
    monkeypatch.setattr(solvers,'solve',lambda *args: pytest.fail('Pure RL called the inner solver'))
    env=UAVMECEnv(control='pure')
    obs=env.reset(5)
    action=np.zeros(146)
    # Saturated route scores assign satellite tasks; unequal shares are evaluated as proposed.
    action[6:106]=np.tile([-1,-1,-1,-1,1],20)
    action[126:146]=-1
    action[126]=1
    new,reward,done,truncated,info=env.step(action)
    assert len(new)==len(obs) and np.isfinite(reward)
    assert info['fallback_count']==0
    assert np.max(info['delay_s'])>1
    assert np.any(np.asarray(info['association'])>0)
    assert info['deadline_violation_rate']>0
    assert info['allocation_invalid_count']==0
    assert not done and not truncated


def test_each_uav_has_its_own_bandwidth_pool():
    env=UAVMECEnv(control='hover_balanced')
    _,_,_,_,info=env.step()
    a=np.asarray(info['association']); b=np.asarray(info['bandwidth_hz'])
    assert info['association_counts']==[0,7,7,6,0]
    for j in (1,2,3):
        assert np.sum(b[a==j])==pytest.approx(10e6)
    assert np.sum(b)==pytest.approx(30e6)
    assert env.cfg.total_uav_bandwidth_hz==pytest.approx(30e6)


def test_task_may_exceed_slot_without_any_repair():
    env=UAVMECEnv(control='pure')
    # CPU time is 1.5s, independent of this one-second control interval.
    data=np.repeat(.5e6,20); cycles=data*500
    allocation=env._simple_resources(np.ones(20,dtype=int))
    allocation['cpu_hz'][:]=cycles/1.5
    env.rho_hz[:]=1e12
    deadline=np.repeat(2.,20)
    a,b,f,delay,raw,invalid=env._evaluate_allocation(allocation,data,cycles,deadline)
    assert np.all(delay>1) and np.all(delay<2)
    np.testing.assert_array_equal(a,np.ones(20,dtype=int))
    np.testing.assert_array_equal(delay,raw)
    assert not np.any(invalid)


def test_zero_rate_is_explicit_failure_not_free_local_execution():
    env=UAVMECEnv(control='pure')
    env.rho_hz[:]=0
    action=np.zeros(146)
    action[6:106]=np.tile([-1,1,-1,-1,-1],20)
    _,reward,_,_,info=env.step(action)
    invalid=np.asarray(info['allocation_invalid_mask'])
    assert invalid.any() and np.isfinite(reward)
    assert np.all(np.asarray(info['association'])[invalid]>0)
    assert all(info['raw_delay_s'][i] is None for i in np.flatnonzero(invalid))
    np.testing.assert_allclose(np.asarray(info['delay_s'])[invalid],
                               100*np.asarray(info['deadline_s'])[invalid])
    assert info['fallback_count']==0 and not info['solver_feasible']
    assert info['deadline_completion']<1


def test_hybrid_passes_individual_deadlines_to_solver(monkeypatch):
    import priority_mec.solvers as solvers
    received={}
    def fake_solve(data,cycles,weights,rho,offset,previous,cfg,deadline_s=None):
        received['deadline']=deadline_s.copy()
        received['weights']=weights.copy()
        return {'association':np.zeros(20,dtype=int),'bandwidth_hz':np.zeros(20),
                'cpu_hz':np.zeros(20),'metrics':{'feasible':False}}
    monkeypatch.setattr(solvers,'solve',fake_solve)
    env=UAVMECEnv(control='hybrid')
    env.scenario.data_bit[0,:]=2e6
    env.scenario.deadline_s[0,:]=.5
    _,reward,_,_,info=env.step()
    np.testing.assert_array_equal(received['deadline'],np.repeat(.5,20))
    np.testing.assert_array_equal(received['weights'],np.repeat(2.,20))
    assert info['delay_s']==[1.]*20
    assert info['deadline_completion']==0
    assert not info['solver_feasible'] and info['fallback_count']==0
    assert reward==pytest.approx(-3.)


def test_common_scenarios_and_restore():
    pure=UAVMECEnv(control='pure')
    local=UAVMECEnv(control='local')
    pure.reset(17); local.reset(17)
    assert pure.scenario.hash==local.scenario.hash
    for _ in range(3):
        _,_,_,_,x=pure.step(np.zeros(146))
        _,_,_,_,y=local.step(np.zeros(6))
        assert x['frame_scenario_hash']==y['frame_scenario_hash']
        np.testing.assert_array_equal(x['next_user_position'],y['next_user_position'])
    state=pure.state_dict()
    clone=UAVMECEnv(control='pure'); clone.load_state_dict(state)
    a=pure.step(np.linspace(-1,1,146));b=clone.step(np.linspace(-1,1,146))
    np.testing.assert_array_equal(a[0],b[0]); assert a[1]==b[1]
    assert a[4]['association']==b[4]['association']
    bad=UAVMECEnv(Config(energy_budget_j=13000),control='pure')
    with pytest.raises(ValueError): bad.load_state_dict(state)


def test_full_episode_energy_and_task_accounting():
    env=UAVMECEnv(control='local')
    total=np.zeros(3)
    for i in range(60):
        _,_,done,_,info=env.step(np.zeros(6))
        total+=info['energy_j']
        assert done==(i==59)
        assert info['association_counts']==[20,0,0,0,0]
        assert info['compute_energy_j']==[0.,0.,0.]
    np.testing.assert_allclose(total,np.repeat(10109.4,3),rtol=1e-14)
    with pytest.raises(RuntimeError):env.step(np.zeros(6))
