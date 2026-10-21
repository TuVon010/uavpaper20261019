"""Association regression checks for feasibility, descent, and honest status."""
from dataclasses import replace
import numpy as np
import pytest
from priority_mec.config import Config
from priority_mec.solvers import solve
from priority_mec.resources import allocate_resources


def problem(n=4, deadline=.3):
    data = np.full(n,1.6e6)
    return data,data*500,np.full(n,1/deadline),np.full((n,4),1e11),np.zeros((n,4))


def test_local_deadline_failure_is_repaired_not_assumed_feasible():
    data,cycle,w,rho,offset=problem()
    out=solve(data,cycle,w,rho,offset,None,Config(),deadline_s=np.full(len(data),.3))
    assert out["solver_feasible"]
    assert np.all(out["association"]>0)
    assert np.all(out["delay_s"]<=.3+1e-7)
    assert out["metrics"]["repair_moves"]==len(data)
    assert not out["metrics"]["global_infeasibility_proven"]


def test_no_one_second_completion_restriction():
    data=np.array([2.4e6]); cycle=data*500
    out=solve(data,cycle,[.5],np.zeros((1,4)),np.zeros((1,4)),None,Config(),
              deadline_s=np.array([2.]))
    assert out["solver_feasible"]
    assert out["association"].tolist()==[0]
    assert out["delay_s"][0]==pytest.approx(1.2)
    assert out["metrics"]["epsilon_equilibrium_certified"]


def test_failed_repair_returns_real_late_local_execution():
    data,cycle,w,rho,offset=problem(3,.3)
    out=solve(data,cycle,w,rho*0,offset,None,Config(),deadline_s=np.full(3,.3))
    assert not out["solver_feasible"]
    assert not out["certified"]
    assert out["status"]=="feasibility_search_failed"
    assert out["association"].tolist()==[0,0,0]
    np.testing.assert_allclose(out["delay_s"],.8)
    np.testing.assert_allclose(out["cpu_hz"],1e9)
    np.testing.assert_allclose(out["bandwidth_hz"],0)
    assert out["metrics"]["deadline_violation_count"]==3
    assert not out["metrics"]["epsilon_equilibrium_certified"]
    assert not out["metrics"]["global_infeasibility_proven"]


def test_independent_node_cache_avoids_repeated_member_set_solves(monkeypatch):
    import priority_mec.solvers as module
    original=module.allocate_resources
    calls=[]
    def track(data,cycle,w,rho,offset,a,cfg,**kwargs):
        calls.append((tuple(a),tuple(data)))
        return original(data,cycle,w,rho,offset,a,cfg,**kwargs)
    monkeypatch.setattr(module,"allocate_resources",track)
    data=np.array([.5e6,.7e6,.9e6,1.1e6])
    out=solve(data,data*500,np.full(4,.8),np.full((4,4),1e11),np.zeros((4,4)),
              np.array([1,2,3,4]),Config())
    assert out["solver_feasible"]
    assert out["metrics"]["cache_hits"]>0
    assert len(calls)==len(set(calls))==out["metrics"]["node_cache_entries"]
    assert all(len(set(a))==1 for a,_ in calls)


def test_descent_and_final_one_user_comparisons():
    data=np.array([.5e6,.8e6,1.2e6])
    cycle=data*500; deadline=np.array([.7,1.1,1.7]); w=1/deadline
    rho=np.array([[1e10,2e9,3e9,1e8],[2e9,1e10,4e9,2e8],[5e9,3e9,1e10,3e8]])
    offset=np.zeros((3,4)); offset[:,3]=.004
    cfg=Config()
    out=solve(data,cycle,w,rho,offset,None,cfg,deadline_s=deadline)
    assert out["solver_feasible"]
    for history in out["metrics"]["descent_histories"]:
        assert np.all(np.diff(history["values"]) < -cfg.solver_epsilon)
    for user in range(3):
        for destination in range(5):
            a=out["association"].copy();a[user]=destination
            candidate=allocate_resources(data,cycle,w,rho,offset,a,cfg,deadline_s=deadline)
            if candidate["feasible"]:
                assert candidate["value"]>=out["value"]-2e-6


def test_capped_scans_do_not_claim_equilibrium():
    data,cycle,w,rho,offset=problem(1,1.)
    cfg=replace(Config(),solver_max_sweeps=1)
    out=solve(data,cycle,w,rho,offset,None,cfg,deadline_s=np.ones(1))
    assert out["solver_feasible"]
    assert out["metrics"]["accepted_moves"]>0
    assert out["metrics"]["descent_histories"][0]["capped"]
    assert not out["metrics"]["epsilon_equilibrium_certified"]


def test_deadline_is_explicit_and_not_inferred_when_supplied():
    data=np.array([1.6e6])
    # Arbitrary objective weight must not accidentally become deadline=1/w.
    out=solve(data,data*500,[10.],np.zeros((1,4)),np.zeros((1,4)),
              None,Config(),deadline_s=np.array([1.5]))
    assert out["solver_feasible"]
    assert out["association"].tolist()==[0]
    with pytest.raises(ValueError):
        solve(data,data*500,[10.],np.zeros((1,4)),np.zeros((1,4)),
              None,Config(),deadline_s=np.array([0.]))


def test_no_energy_term_changes_association():
    args=problem(4,1.)
    first=solve(*args,None,Config())
    altered=replace(Config(),energy_budget_j=1.,uav_joule_per_cycle=1.)
    second=solve(*args,None,altered)
    np.testing.assert_array_equal(first["association"],second["association"])
    np.testing.assert_allclose(first["delay_s"],second["delay_s"])



def test_bounded_two_user_repair_can_release_a_congested_node(monkeypatch):
    import priority_mec.solvers as module
    def limited_node(data,cycles,w,rho,offset,a,cfg,deadline_s=None):
        node=int(a[0]); users=np.rint(np.asarray(data)/1e5).astype(int)
        ok=(len(data)==1 and rho[0,node-1]>0)
        delay=np.full(len(data),.9)
        if ok:
            delay[:]=.2 if node==2 else .1
        value=float(np.asarray(w)@delay)
        return dict(association=np.asarray(a),bandwidth_hz=np.full(len(data),1e6/len(data)),
                    cpu_hz=np.full(len(data),1e9/len(data)),delay_s=delay,value=value,
                    lower_bound=value,gap=0.,status='optimal' if ok else 'infeasible',
                    feasible=bool(ok),certified=bool(ok),certified_infeasible=not ok,
                    metrics={})
    monkeypatch.setattr(module,'allocate_resources',limited_node)
    data=np.array([1.6e6,1.7e6]); cycle=np.full(2,.8e9)
    rho=np.array([[1e12,1e12,0,0],[1e12,0,0,0]])
    deadline=np.array([.4,.5])
    out=solve(data,cycle,1/deadline,rho,np.zeros((2,4)),None,Config(),deadline_s=deadline)
    assert out['solver_feasible']
    assert out['association'].tolist()==[2,1]
    assert out['metrics']['repair_pair_moves']==1
    assert out['metrics']['repair_trials']<=2*4+80
