"""Slot-boundary block channels and analytic straight-line flight safety."""
import numpy as np
from scipy.optimize import minimize

LIGHT_MPS = 299792458.0

# 3GPP TR 38.811 V15.3.0, Tables 6.6.1-1, 6.6.2-2 and 6.7.2-3a.
# Outdoor Urban S-band is a proxy for a damaged urban outdoor deployment; these
# are not measured rubble-specific coefficients. Nearest reference angle is used.
SATELLITE_CHANNEL_MODEL = '3gpp_38811_urban_sband_effective_block_v1'
_SATELLITE_LOS_PROBABILITY = np.array([.246, .386, .493, .613, .726, .805, .919, .968, .992])
_SATELLITE_NLOS_CLUTTER_DB = np.array([34.3, 30.9, 29.0, 27.7, 26.8, 26.2, 25.8, 25.5, 25.5])
_SATELLITE_LOS_K_MEAN_DB = np.array([31.83, 18.78, 10.49, 7.46, 6.52, 5.47, 4.54, 4.03, 3.68])


def satellite_urban_parameters(elevation_deg):
    """S-band outdoor-urban tables with nearest-10-degree reference selection.

    K is fixed to the table's dB mean at a given elevation, rather than drawing
    another uncertain large-scale parameter. LOS/NLOS shadow standard deviations
    are 4/6 dB. Values outside the table are clamped; link availability is handled
    separately by the configured minimum elevation.
    """
    index = np.clip(np.floor(np.asarray(elevation_deg)/10+.5).astype(int)-1, 0, 8)
    return (_SATELLITE_LOS_PROBABILITY[index], _SATELLITE_NLOS_CLUTTER_DB[index],
            _SATELLITE_LOS_K_MEAN_DB[index])


def satellite_channel_components(elevation_deg, los_uniform, shadow_normal, fade_normal):
    """Effective per-slot SISO gain factors, not a full wideband CDL simulation.

    Large-scale loss comprises clutter and a zero-mean normal loss in dB;
    small-scale power has conditional mean one (Rician LOS, Rayleigh NLOS).
    100 MHz denotes an abstract pooled resource. A scalar effective channel does
    not assert that an actual 100 MHz waveform is physically frequency-flat.
    """
    p_los, nlos_clutter, k_db = satellite_urban_parameters(elevation_deg)
    los = np.asarray(los_uniform) < p_los
    clutter_db = np.where(los, 0.0, nlos_clutter)
    shadow_db = np.asarray(shadow_normal)*np.where(los, 4.0, 6.0)
    z = np.asarray(fade_normal)
    k = 10**(k_db/10)
    rice_power = ((np.sqrt(k/(k+1))+z[..., 0]/np.sqrt(2*(k+1)))**2
                  + z[..., 1]**2/(2*(k+1)))
    rayleigh_power = np.sum(z*z, axis=-1)/2
    return {'satellite_los': los, 'satellite_los_probability': p_los,
            'satellite_clutter_loss_db': clutter_db, 'satellite_shadow_loss_db': shadow_db,
            'satellite_fading_power': np.where(los, rice_power, rayleigh_power),
            # Tabulated LOS K; not active for a NLOS Rayleigh sample.
            'satellite_rician_k_db': k_db}


def propulsion_power(speed_mps, cfg):
    """Monotone upper approximation, including the full positive hover power."""
    v = np.asarray(speed_mps)
    c3 = .5 * cfg.rotor_drag_ratio * cfg.air_density * cfg.rotor_solidity * cfg.rotor_area_m2
    return cfg.rotor_p0_w*(1+3*v*v/cfg.rotor_tip_mps**2)+cfg.rotor_pi_w+c3*v**3


def initial_uav_positions(cfg):
    theta = np.arange(cfg.num_uavs) * 2*np.pi/cfg.num_uavs
    return cfg.area_m/2 + cfg.area_m/2*np.stack((np.cos(theta), np.sin(theta)), axis=1)


def minimum_separation(position, velocity, dt):
    minimum = np.inf
    for i in range(len(position)):
        for j in range(i):
            r, v = position[i]-position[j], velocity[i]-velocity[j]
            vv = np.dot(v, v)
            tau = np.clip(-np.dot(r,v)/vv, 0, dt) if vv > 0 else 0
            minimum = min(minimum, float(np.linalg.norm(r+tau*v)))
    return minimum


def safe_velocity(position, requested, cfg):
    """Bound requests then minimally repair conflicts using a convex QCQP.

    Collision-free requests return without numerical optimization. For conflicting
    requests the objective is squared deviation from the speed/boundary-bounded
    request. Fixed pair normals impose n_ij.(v_i-v_j)>=(D-d_ij)/dt. Their linear
    interpolation proves continuous separation throughout the constant-speed
    slot. Speed circles make this a convex QCQP, not a purely linear-constraint
    QP. The sufficient pair halfspaces may conservatively exclude safe turns.
    Numerical failures return the feasible zero velocity and are reported.
    safety_scale is only a norm ratio; velocities no longer share one scale.
    """
    position = np.asarray(position, dtype=float)
    v = np.asarray(requested, dtype=float).reshape(cfg.num_uavs, 2).copy()
    dt, vmax = cfg.frame_s, cfg.speed_max_mps
    if position.shape != v.shape or np.any(~np.isfinite(position)) or np.any(~np.isfinite(v)):
        raise ValueError('Flight inputs must be finite [J,2] arrays')
    if (np.any(position < -1e-8) or np.any(position > cfg.area_m+1e-8)
            or minimum_separation(position, np.zeros_like(v), dt) < cfg.safe_distance_m-1e-8):
        raise ValueError('Safety projection requires an initially safe in-map state')
    norm = np.linalg.norm(v, axis=1)
    v *= np.minimum(1, vmax/np.maximum(norm, 1e-30))[:, None]
    capped_request = v.copy()
    endpoint = position+dt*v
    boundary = bool(np.any(endpoint < 0) or np.any(endpoint > cfg.area_m))
    collision = bool(minimum_separation(position, v, dt) < cfg.safe_distance_m-1e-10)
    projected_endpoint = np.clip(endpoint, 0, cfg.area_m)
    boundary_projected = np.any(projected_endpoint != endpoint, axis=1)
    v = (projected_endpoint-position)/dt
    projected_collision = bool(minimum_separation(position, v, dt) < cfg.safe_distance_m-1e-10)
    target = v.copy()
    success, status, iterations = True, 'not_needed', 0
    if projected_collision:
        rows, limits = [], []
        for i in range(cfg.num_uavs):
            for j in range(i):
                difference = position[i]-position[j]
                distance = float(np.linalg.norm(difference))
                row = np.zeros_like(v)
                row[i], row[j] = difference/distance, -difference/distance
                rows.append(row.ravel())
                limits.append((cfg.safe_distance_m-distance)/(dt*vmax))
        matrix, rhs = np.asarray(rows), np.asarray(limits)
        desired = (target/vmax).ravel()
        lower = np.maximum(-1., -position/(dt*vmax)).ravel()
        upper = np.minimum(1., (cfg.area_m-position)/(dt*vmax)).ravel()

        def circle_jacobian(x):
            derivative = np.zeros((cfg.num_uavs, 2*cfg.num_uavs))
            for k in range(cfg.num_uavs):
                derivative[k, 2*k:2*k+2] = -2*x[2*k:2*k+2]
            return derivative

        constraints = [
            {'type': 'ineq', 'fun': lambda x: matrix@x-rhs, 'jac': lambda x: matrix},
            {'type': 'ineq', 'fun': lambda x: 1.-np.sum(x.reshape(-1, 2)**2, axis=1),
             'jac': circle_jacobian}]
        try:
            result = minimize(lambda x: .5*np.sum((x-desired)**2), np.zeros(v.size),
                              jac=lambda x: x-desired, bounds=list(zip(lower, upper)),
                              constraints=constraints, method='SLSQP',
                              options={'ftol': 1e-12, 'maxiter': 100, 'disp': False})
            candidate = np.asarray(result.x).reshape(v.shape)*vmax
            iterations = int(getattr(result, 'nit', 0))
            final = position+dt*candidate
            checked = (np.all(np.isfinite(candidate))
                       and np.all(np.linalg.norm(candidate, axis=1) <= vmax+1e-8)
                       and np.all(final >= -1e-8) and np.all(final <= cfg.area_m+1e-8)
                       and np.all(matrix@result.x-rhs >= -1e-10)
                       and minimum_separation(position, candidate, dt) >= cfg.safe_distance_m-1e-10)
            success = bool(result.success and checked)
            status = 'optimal' if success else 'solver_or_postcheck_failure'
            v = candidate if success else np.zeros_like(v)
        except (ValueError, FloatingPointError, RuntimeError):
            success, status = False, 'solver_exception'
            v = np.zeros_like(v)
    target_norm = float(np.linalg.norm(target))
    scale = float(np.linalg.norm(v)/target_norm) if target_norm > 0 else 1.0
    return v, {'boundary_attempt': boundary, 'collision_attempt': collision,
               'boundary_projected_uavs': boundary_projected.tolist(),
               'projected_collision': projected_collision, 'safety_scale': scale,
               'safety_method': 'fixed_normal_minimum_deviation_qcqp_v1',
               'projection_method': 'fixed_normal_minimum_deviation_qcqp_v1',
               'projection_called': projected_collision,
               'projection_failed': not success,
               'projection_status': status, 'projection_iterations': iterations,
               'projection_correction_norm_mps': np.linalg.norm(v-capped_request, axis=1).tolist(),
               'minimum_separation_m': minimum_separation(position, v, dt),
               'requested_speed_mean_mps': float(np.mean(norm)),
               'actual_speed_mean_mps': float(np.mean(np.linalg.norm(v, axis=1)))}


def satellite_geometry(user_position, t_s, cfg):
    """Circular equatorial orbit with closest overhead pass at episode midpoint."""
    earth, orbit = cfg.earth_radius_m, cfg.earth_radius_m + cfg.satellite_altitude_m
    angular = np.sqrt(cfg.earth_mu_m3_s2/orbit**3) - cfg.earth_spin_rad_s
    angle = angular*(t_s-cfg.episode_frames*cfg.frame_s/2)
    satellite = np.array([orbit*np.cos(angle), orbit*np.sin(angle), 0.0])
    xy = np.asarray(user_position)-cfg.area_m/2
    longitude, latitude = xy[:, 0]/earth, xy[:, 1]/earth
    ground = earth*np.column_stack((np.cos(latitude)*np.cos(longitude),
                                   np.cos(latitude)*np.sin(longitude), np.sin(latitude)))
    difference = satellite[None, :] - ground
    distance = np.linalg.norm(difference, axis=1)
    elevation = np.arcsin(np.clip(np.sum(difference*(ground/earth), axis=1)/distance, -1, 1))
    return distance, np.rad2deg(elevation)


def channel_parameters(user_position, uav_position, frame, scenario, cfg):
    """Returns P*g/N0 in Hz. Current observed block is frozen during this slot.

    Geometry is sampled at the slot START. Motion controls the next slot's channel;
    it cannot obtain a free full-slot rate at the end position. A 1-second equivalent
    fading block is a simulation abstraction, not a claimed physical coherence time.
    """
    displacement = np.asarray(user_position)[:, None, :] - np.asarray(uav_position)[None, :, :]
    horizontal = np.linalg.norm(displacement, axis=-1)
    distance = np.sqrt(horizontal**2+cfg.altitude_m**2)
    elevation = np.rad2deg(np.arctan2(cfg.altitude_m, horizontal))
    p_los = 1/(1+cfg.los_a*np.exp(-cfg.los_b*(elevation-cfg.los_a)))
    los = scenario.los_uniform[frame] < p_los
    free_loss = 20*np.log10(4*np.pi*distance*cfg.carrier_hz/LIGHT_MPS)
    excess = np.where(los, cfg.los_extra_db, cfg.nlos_extra_db)
    z = scenario.fade_normal[frame]
    k = 10**(cfg.rician_k_db/10)
    rice = (np.sqrt(k/(k+1))+z[...,0]/np.sqrt(2*(k+1)))**2 + z[...,1]**2/(2*(k+1))
    rayleigh = np.sum(z*z, axis=-1)/2
    fading = np.where(los, rice, rayleigh)
    gain = 10**(-(free_loss+excess)/10)*fading
    noise_w_hz = 10**((cfg.noise_dbm_hz+cfg.noise_figure_db-30)/10)
    rho_ground = cfg.tx_power_w*gain/noise_w_hz
    sat_distance, sat_elevation = satellite_geometry(user_position, frame*cfg.frame_s, cfg)
    sat_fspl = 20*np.log10(4*np.pi*sat_distance*cfg.satellite_carrier_hz/LIGHT_MPS)
    sat_channel = satellite_channel_components(
        sat_elevation, scenario.satellite_los_uniform[frame],
        scenario.satellite_shadow_normal[frame], scenario.satellite_fade_normal[frame])
    sat_large_scale_loss = (sat_fspl + cfg.satellite_extra_loss_db
                           + sat_channel['satellite_clutter_loss_db']
                           + sat_channel['satellite_shadow_loss_db'])
    eirp_dbw = 10*np.log10(cfg.tx_power_w)
    cn0 = eirp_dbw + cfg.satellite_gt_db_k-sat_large_scale_loss+228.6
    rho_sat = np.where(sat_elevation >= cfg.satellite_min_elevation_deg,
                       10**(cn0/10)*sat_channel['satellite_fading_power'], 0.0)
    rho = np.column_stack((rho_ground, rho_sat))
    offset = np.zeros_like(rho)
    offset[:, -1] = 2*sat_distance/LIGHT_MPS
    return rho, offset, {'los': los, 'satellite_elevation_deg': sat_elevation,
                         'satellite_distance_m': sat_distance,
                         'satellite_fspl_db': sat_fspl,
                         'satellite_large_scale_loss_db': sat_large_scale_loss,
                         'satellite_channel_model': SATELLITE_CHANNEL_MODEL, **sat_channel}


def link_rate(bandwidth_hz, rho_hz):
    b, rho = np.broadcast_arrays(np.asarray(bandwidth_hz, dtype=float), np.asarray(rho_hz, dtype=float))
    rate = np.zeros_like(b)
    positive = b > 0
    rate[positive] = b[positive]*np.log1p(rho[positive]/b[positive])/np.log(2)
    return rate
