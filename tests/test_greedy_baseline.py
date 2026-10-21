"""Check the specified coarse-independent/singleton-equal-share comparator."""
from dataclasses import replace
import inspect
import json
from unittest.mock import patch
import numpy as np
import pytest
from priority_mec.config import Config
from priority_mec.environment import UAVMECEnv
from priority_mec.greedy_baseline import greedy_allocation, plan_velocity
from priority_mec.physics import initial_uav_positions, minimum_separation, safe_velocity, link_rate


def tasks(c):
    data=np.linspace(.5e6,2e6,c.num_users)
    work=data*c.cycles_per_bit
    deadline=np.linspace(.5,2.,c.num_users)[::-1]
    rho=np.full((c.num_users,c.num_uavs+1),2e9)
    return data,work,deadline,rho,np.zeros_like(rho)


def test_singleton_routing_equal_shares_and_real_congestion():
    c=Config();data,work,deadline,rho,offset=tasks(c)
    result=greedy_allocation(data,work,deadline,rho,offset,c)
    # Make the prescribed decision independently, with full resource at each node.
    bandwidth=np.array([c.uav_bandwidth_hz]*3+[c.satellite_bandwidth_hz])
    frequency=np.array([c.uav_cpu_hz]*3+[c.satellite_cpu_hz])
    estimated=np.column_stack((work/c.local_cpu_hz,
        data[:,None]/link_rate(bandwidth[None],rho)+work[:,None]/frequency[None]+offset))
    np.testing.assert_array_equal(result['association'],estimated.argmin(axis=1))
    assert result['metrics']['order']==list(range(19,-1,-1))
    for node in range(1,5):
        selected=result['association']==node
        if selected.any():
            np.testing.assert_allclose(result['bandwidth_hz'][selected],bandwidth[node-1]/selected.sum())
            np.testing.assert_allclose(result['cpu_hz'][selected],frequency[node-1]/selected.sum())
    initial=np.array(result['metrics']['initial_estimated_delay_s'])
    final=np.array(result['metrics']['final_estimated_delay_s'])
    assert np.all(final>=initial-1e-12)
    assert np.any(final>initial+.1)
    assert not result['metrics']['spare_return'] and not result['metrics']['congestion_repair']
    assert result['metrics']['allocation_iterations']==1


def test_deadline_sort_does_not_change_singleton_routes_or_equal_shares():
    c=Config();data,work,deadline,rho,offset=tasks(c)
    first=greedy_allocation(data,work,deadline,rho,offset,c)
    second=greedy_allocation(data,work,deadline[::-1],rho,offset,c)
    for key in ('association','bandwidth_hz','cpu_hz'):
        np.testing.assert_array_equal(first[key],second[key])
    assert first['metrics']['order']!=second['metrics']['order']


def test_blocked_links_fallback_local_and_json_metrics_finite():
    c=Config();data,work,deadline,rho,offset=tasks(c)
    result=greedy_allocation(data,work,deadline,np.zeros_like(rho),offset,c)
    np.testing.assert_array_equal(result['association'],0)
    np.testing.assert_array_equal(result['bandwidth_hz'],0)
    np.testing.assert_array_equal(result['cpu_hz'],0)
    np.testing.assert_allclose(result['metrics']['final_estimated_delay_s'],work/c.local_cpu_hz)
    json.dumps(result['metrics'],allow_nan=False)


def test_shared_deadline_failures_are_not_repaired_or_hidden():
    c=Config();data,work,deadline,rho,offset=tasks(c)
    # Satellite appears fastest alone; twenty colocated tasks overload its shares.
    rho[:,:3]=0.;deadline[:]=.05
    result=greedy_allocation(data,work,deadline,rho,offset,c)
    np.testing.assert_array_equal(result['association'],4)
    assert result['metrics']['actual_deadline_violation_count']==20
    assert np.all(np.array(result['metrics']['final_estimated_delay_s'])>deadline)


def test_grid_has_49_distinct_actions_without_refinement_or_optimizer():
    c=Config();users=np.full((20,2),500.);position=initial_uav_positions(c)
    rho=np.full((20,4),1e8)
    with patch('priority_mec.physics.safe_velocity',side_effect=AssertionError('Projection belongs to environment')):
        velocity,info=plan_velocity(users,position,np.ones(20),rho,c)
    assert info['distinct_candidates_per_uav']==49
    assert info['nonzero_candidates_per_uav']==48
    assert info['candidates_evaluated']==147 and info['refine_rounds']==0
    assert info['speed_levels_mps']==[0.,5.,10.,20.]
    assert np.all(np.array(info['score_after'])>=np.array(info['score_before'])-1e-5)
    for speed in np.linalg.norm(velocity,axis=1):
        assert np.min(np.abs(np.array([0.,5.,10.,20.])-speed))<1e-10


def test_motion_selection_independent_of_other_uavs_and_permutation():
    c=Config();users=np.full((20,2),500.);position=initial_uav_positions(c)
    rho=np.full((20,4),1e8);weight=np.linspace(.5,2.,20)
    first,_=plan_velocity(users,position,weight,rho,c)
    changed=position.copy();changed[1:]=[[999.,999.],[500.,100.]]
    second,_=plan_velocity(users,changed,weight,rho,c)
    np.testing.assert_array_equal(first[0],second[0])
    order=np.array([2,0,1]);permuted_rho=rho.copy();permuted_rho[:,:3]=rho[:,order]
    permuted,_=plan_velocity(users,position[order],weight,permuted_rho,c)
    np.testing.assert_array_equal(first[order],permuted)


def test_environment_repairs_raw_collision_after_independent_selection():
    c=Config();env=UAVMECEnv(c,seed=17,control='greedy_rate')
    env.uav_position=np.array([[475.,500.],[525.,500.],[999.,999.]])
    env.user_position[:]=500.;env._refresh_channel()
    raw,info=plan_velocity(env.user_position,env.uav_position,1/env.scenario.deadline_s[0],env.rho_hz,c)
    assert minimum_separation(env.uav_position,raw,1.)<50.
    initial=env.uav_position.copy()
    _,_,_,_,result=env.step()
    assert result['collision_attempt']
    np.testing.assert_array_equal(result['requested_velocity_mps'],raw)
    assert minimum_separation(initial,np.array(result['uav_velocity']),1.)>=50.-1e-7
    assert not result['projection_failed']


def test_no_hidden_rng_or_energy_inputs_and_deterministic_hover_tie():
    c=Config();users=np.full((20,2),500.);position=initial_uav_positions(c)
    rho=np.zeros((20,4));state=np.random.get_state()
    velocity,first=plan_velocity(users,position,np.ones(20),rho,c)
    np.testing.assert_array_equal(velocity,0)
    changed=replace(c,energy_budget_j=.001,energy_penalty=1e9,rotor_p0_w=1e12)
    other,second=plan_velocity(users,position,np.ones(20),rho,changed)
    np.testing.assert_array_equal(velocity,other);assert first==second
    for before,after in zip(state,np.random.get_state()):np.testing.assert_array_equal(before,after)


def test_no_association_solver_or_kkt_import_and_bad_inputs_fail():
    import priority_mec.greedy_baseline as module
    assert 'from .solvers' not in inspect.getsource(module)
    assert 'from .resources' not in inspect.getsource(module)
    c=Config();data,work,deadline,rho,offset=tasks(c)
    with pytest.raises(ValueError):greedy_allocation(data,work,deadline,-rho,offset,c)
    with pytest.raises(ValueError):plan_velocity(np.zeros((20,2)),initial_uav_positions(c),np.ones(20),rho,c,direction_count=0)
    with pytest.raises(ValueError):plan_velocity(np.zeros((20,2)),initial_uav_positions(c),np.ones(20),rho,c,refine_rounds=1)
