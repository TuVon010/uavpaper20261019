"""Exogenous keyed randomness, independent of controller and solver call counts."""
import hashlib
import random
import numpy as np


def keyed_rng(seed, episode, slot, stream):
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(episode), int(slot), int(stream)]))


def reflect_positions(position, area_m):
    """Mirror at the map boundary; preserves every signed speed sample."""
    return area_m - np.abs(np.mod(position, 2 * area_m) - area_m)


def gauss_markov_update(speed, angle, mean_speed, mean_angle, speed_noise, angle_noise, cfg):
    a, b = cfg.user_speed_alpha, cfg.user_direction_alpha
    next_speed = a * speed + (1-a) * mean_speed + np.sqrt(1-a*a) * speed_noise
    next_angle = b * angle + (1-b) * mean_angle + np.sqrt(1-b*b) * angle_noise
    return next_speed, next_angle


class Scenario:
    def __init__(self, cfg, seed=1, episode=0):
        self.cfg, self.seed, self.episode = cfg, int(seed), int(episode)
        n, h = cfg.num_users, cfg.episode_frames
        # The requested Python random.random semantics, isolated from global state.
        python_seed = int.from_bytes(hashlib.sha256(f'{seed}:{episode}:mean_speed'.encode()).digest()[:8], 'little')
        py = random.Random(python_seed)
        self.mean_speed = np.asarray([py.random() for _ in range(n)])
        self.mean_angle = keyed_rng(seed, episode, 0, 1).uniform(0, 2*np.pi, n)
        center=cfg.area_m*np.array([cfg.user_cluster_center_x_fraction,cfg.user_cluster_center_y_fraction])
        self.initial_position = keyed_rng(seed, episode, 0, 2).uniform(
            center-cfg.user_cluster_halfwidth_m,center+cfg.user_cluster_halfwidth_m,(n,2))
        self.data_bit = np.empty((h, n))
        self.deadline_s = np.empty((h, n))
        self.speed_noise = np.empty((h, n))
        self.angle_noise = np.empty((h, n))
        self.los_uniform = np.empty((h, n, cfg.num_uavs))
        self.fade_normal = np.empty((h, n, cfg.num_uavs, 2))
        self.route_uniform = np.empty((h, n))
        # Isolated streams preserve all existing task, mobility and UAV draws.
        self.satellite_los_uniform = np.empty((h, n))
        self.satellite_shadow_normal = np.empty((h, n))
        self.satellite_fade_normal = np.empty((h, n, 2))
        for t in range(h):
            self.data_bit[t] = keyed_rng(seed, episode, t, 10).uniform(cfg.data_min_bit, cfg.data_max_bit, n)
            self.deadline_s[t] = keyed_rng(seed, episode, t, 11).uniform(cfg.deadline_min_s, cfg.deadline_max_s, n)
            self.speed_noise[t] = keyed_rng(seed, episode, t, 12).normal(0, np.sqrt(cfg.user_speed_noise_variance), n)
            self.angle_noise[t] = keyed_rng(seed, episode, t, 13).normal(0, cfg.user_direction_noise_std_rad, n)
            self.los_uniform[t] = keyed_rng(seed, episode, t, 14).random((n, cfg.num_uavs))
            self.fade_normal[t] = keyed_rng(seed, episode, t, 15).normal(size=(n, cfg.num_uavs, 2))
            self.route_uniform[t] = keyed_rng(seed, episode, t, 16).random(n)
            self.satellite_los_uniform[t] = keyed_rng(seed, episode, t, 17).random(n)
            self.satellite_shadow_normal[t] = keyed_rng(seed, episode, t, 18).normal(size=n)
            self.satellite_fade_normal[t] = keyed_rng(seed, episode, t, 19).normal(size=(n, 2))
        self.hash = self._hash()

    def _hash(self):
        digest = hashlib.sha256()
        for x in (self.mean_speed, self.mean_angle, self.initial_position, self.data_bit,
                  self.deadline_s, self.speed_noise, self.angle_noise, self.los_uniform,
                  self.fade_normal, self.route_uniform, self.satellite_los_uniform,
                  self.satellite_shadow_normal, self.satellite_fade_normal):
            digest.update(np.asarray(x, dtype='<f8').tobytes())
        return digest.hexdigest()

    def frame_hash(self, frame):
        d = hashlib.sha256()
        for x in (self.data_bit[frame], self.deadline_s[frame], self.speed_noise[frame],
                  self.angle_noise[frame], self.los_uniform[frame], self.fade_normal[frame],
                  self.route_uniform[frame], self.satellite_los_uniform[frame],
                  self.satellite_shadow_normal[frame], self.satellite_fade_normal[frame]):
            d.update(np.asarray(x, dtype='<f8').tobytes())
        return d.hexdigest()
