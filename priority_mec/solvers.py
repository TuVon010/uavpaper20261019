"""Deadline-constrained common-cost association with independent node caches."""
from __future__ import annotations
import time
import numpy as np
from .resources import allocate_resources


def solve(data_bit, cycles, weights, rho_hz, offset_s,
          previous_association, cfg, deadline_s=None):
    """Feasible seed repair followed by strictly decreasing potential moves.

    Seed repair failure is explicit and is not a global infeasibility proof.
    A late local task stays visible; no task is dropped or freely repaired.
    """
    started = time.perf_counter()
    data, cycle, weight = [np.asarray(x, float) for x in (data_bit, cycles, weights)]
    rho, offset = np.asarray(rho_hz, float), np.asarray(offset_s, float)
    n, m = len(data), int(getattr(cfg, "num_uavs", getattr(cfg, "n_uavs", 3)))
    deadline = 1. / weight if deadline_s is None else np.asarray(deadline_s, float)
    eps, tol = float(cfg.solver_epsilon), float(getattr(cfg, "feasibility_tol", 1e-7))
    if (data.ndim != 1 or any(a.shape != (n,) for a in (cycle, weight, deadline))
            or rho.shape != (n, m+1) or offset.shape != rho.shape):
        raise ValueError("incompatible task/deadline/link shapes")
    if any(not np.isfinite(a).all() or np.any(a <= 0) for a in (data, cycle, weight, deadline)):
        raise ValueError("positive finite tasks, weights, and deadlines required")
    if int(cfg.solver_max_sweeps) < 1:
        raise ValueError("solver_max_sweeps must be positive")
    cache, bounds = {}, {}
    counts = dict(resource_solves=0, cache_hits=0, pruned_candidates=0,
                  unresolved_candidates=0, accepted_moves=0, scans=0,
                  repair_trials=0, repair_moves=0, repair_pair_moves=0)
    local_delay = cycle / float(cfg.local_cpu_hz)
    local_ok, order = local_delay <= deadline + tol, np.argsort(-weight, kind="stable")

    def lower_bound(node, users):
        key = node, tuple(int(i) for i in users)
        if key in bounds: return bounds[key]
        if not len(users): value = 0.
        elif node == 0: value = float(weight[users] @ local_delay[users])
        else:
            r = rho[users, node-1]
            if np.any(r <= 0) or not np.isfinite(r).all(): value = np.inf
            else:
                b = float(cfg.satellite_bandwidth_hz if node == m+1 else cfg.uav_bandwidth_hz)
                f = float(cfg.satellite_cpu_hz if node == m+1 else cfg.uav_cpu_hz)
                rate = b * np.log1p(r.sum()/b) / np.log(2.)
                bw = max(np.sqrt(weight[users]*data[users]).sum()**2 / rate,
                         float(np.sum(weight[users]*data[users]*np.log(2.)/r)))
                value = float(weight[users] @ offset[users,node-1]
                              + np.sqrt(weight[users]*cycle[users]).sum()**2/f + bw)
        bounds[key] = value
        return value

    def allocated(node, users):
        key = node, tuple(int(i) for i in users)
        if key in cache:
            counts["cache_hits"] += 1
            return cache[key]
        allocation = allocate_resources(data[users], cycle[users], weight[users],
            rho[users], offset[users], np.full(len(users), node, int), cfg,
            deadline_s=deadline[users])
        counts["resource_solves"] += 1
        cache[key] = allocation
        return allocation

    def evaluate(association, incumbent=np.inf, repairing=False):
        a = np.asarray(association, int)
        groups = [np.flatnonzero(a == j) for j in range(m+2)]
        lower = sum(lower_bound(j, g) for j, g in enumerate(groups))
        if np.isfinite(incumbent) and lower >= incumbent-eps:
            counts["pruned_candidates"] += 1
            return dict(status="pruned", lower_bound=lower)
        delay, bw, cpu = local_delay.copy(), np.zeros(n), np.full(n,float(cfg.local_cpu_hz))
        local_bad = groups[0][~local_ok[groups[0]]]
        remote_ok, remote_cert, numeric = True, True, False
        proven = bool(len(local_bad) and not repairing)
        lower = float(weight[groups[0]] @ local_delay[groups[0]])
        diagnostics = []
        for node in range(1,m+2):
            users = groups[node]
            if not len(users): continue
            result = allocated(node,users)
            delay[users], bw[users], cpu[users] = result["delay_s"],result["bandwidth_hz"],result["cpu_hz"]
            lower += float(result["lower_bound"])
            remote_ok &= bool(result["feasible"])
            remote_cert &= bool(result["certified"])
            numeric |= result["status"] == "unresolved"
            proven |= bool(result.get("certified_infeasible",False))
            diagnostics.append(dict(node=node,users=users.tolist(),status=result["status"],
                                    gap=float(result["gap"]),metrics=result["metrics"]))
        feasible = bool(remote_ok and np.all(delay <= deadline+tol))
        certified = bool(feasible and remote_cert)
        value = float(weight @ delay)
        return dict(association=a.copy(),bandwidth_hz=bw,cpu_hz=cpu,delay_s=delay,
                    value=value,lower_bound=lower,gap=max(0.,value-lower),
                    feasible=feasible,certified=certified,remote_feasible=bool(remote_ok),
                    node_diagnostics=diagnostics,
                    status="optimal" if certified else "infeasible" if proven else
                           "unresolved" if numeric else "deadline_violated")

    def repair():
        current = evaluate(np.zeros(n,int),repairing=True)
        pair_used, budget = 0,int(getattr(cfg,"solver_repair_pair_budget",80))
        for user in order:
            if local_ok[user] or current["association"][user] != 0: continue
            selected = None
            for destination in range(1,m+2):
                a = current["association"].copy(); a[user] = destination
                result = evaluate(a,repairing=True); counts["repair_trials"] += 1
                if result["remote_feasible"] and (selected is None or result["value"] < selected["value"]):
                    selected = result
            # Bounded two-user relocation repairs congestion without 5**N search.
            if selected is None and pair_used < budget:
                for destination in range(1,m+2):
                    occupants = np.flatnonzero(current["association"] == destination)
                    occupants = sorted(occupants,key=lambda i:(not local_ok[i],weight[i],int(i)))
                    for other in occupants:
                        for alternate in range(m+2):
                            if alternate == destination or (alternate == 0 and not local_ok[other]): continue
                            if pair_used >= budget: break
                            a = current["association"].copy()
                            a[user],a[other] = destination,alternate
                            result = evaluate(a,repairing=True)
                            pair_used += 1; counts["repair_trials"] += 1
                            if result["remote_feasible"] and (selected is None or result["value"] < selected["value"]):
                                selected = result
                        if pair_used >= budget: break
                    if pair_used >= budget: break
                if selected is not None: counts["repair_pair_moves"] += 1
            if selected is not None:
                current = selected; counts["repair_moves"] += 1
        return current

    starts = []
    if previous_association is not None:
        previous = np.asarray(previous_association,int)
        if previous.shape == (n,) and np.all((previous>=0)&(previous<=m+1)):
            warm = evaluate(previous)
            if warm["feasible"]: starts.append(warm)
    repaired = repair()
    if repaired["feasible"] and not any(np.array_equal(repaired["association"],x["association"]) for x in starts):
        starts.append(repaired)
    best = min(starts,key=lambda x:x["value"]) if starts else repaired
    equilibrium,histories = False,[]
    for initial in starts:
        current,history,certified_scan,capped = initial,[float(initial["value"])],False,True
        for _ in range(int(cfg.solver_max_sweeps)):
            improved,unresolved = False,not current["certified"]
            counts["scans"] += 1
            for user in order:
                selected = current
                for destination in range(m+2):
                    if destination == current["association"][user] or (destination==0 and not local_ok[user]): continue
                    a = current["association"].copy(); a[user] = destination
                    candidate = evaluate(a,selected["value"])
                    if candidate["status"] in ("pruned","infeasible"): continue
                    if candidate["feasible"] and candidate["value"] < selected["lower_bound"]-eps: selected = candidate
                    if candidate["lower_bound"] < selected["value"]-eps:
                        unresolved = True
                        if not candidate["certified"]: counts["unresolved_candidates"] += 1
                if selected is not current:
                    current = selected; history.append(float(current["value"]))
                    improved = True; counts["accepted_moves"] += 1
            if not improved:
                capped = False; certified_scan = bool(not unresolved and current["certified"]); break
        histories.append(dict(values=history,capped=capped,epsilon_equilibrium_certified=certified_scan))
        if current["value"] < best["value"]: best,equilibrium = current,certified_scan
        elif np.array_equal(current["association"],best["association"]): equilibrium |= certified_scan
    answer = {k:best[k] for k in ("association","bandwidth_hz","cpu_hz","delay_s","value")}
    answer.update(feasible=bool(best["feasible"]),certified=bool(best["certified"]),
                  solver_feasible=bool(best["feasible"]),
                  status="feasible" if best["feasible"] else "feasibility_search_failed")
    answer["metrics"] = dict(**counts,node_cache_entries=len(cache),
        epsilon_equilibrium_certified=bool(equilibrium),equilibrium_tolerance=eps,
        resource_gap=float(best["gap"]),resource_lower_bound=float(best["lower_bound"]),
        resource_certified=bool(best["certified"]),feasible=bool(best["feasible"]),
        feasibility_search_failed=not best["feasible"],global_infeasibility_proven=False,
        deadline_violation_count=int(np.count_nonzero(best["delay_s"]>deadline+tol)),
        max_deadline_violation_s=float(np.maximum(best["delay_s"]-deadline,0).max(initial=0)),
        node_diagnostics=best["node_diagnostics"],descent_histories=histories,
        wall_s=time.perf_counter()-started)
    return answer


