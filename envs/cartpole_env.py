"""
CartPole environment adapted for the Residual RL pixel-based pipeline.

This wraps the classic CartPole physics with:
- Continuous action space (Box(-1, 1))
- Dictionary observations: {'image': ..., 'state': ...}
- Offscreen rendering via pygame (no display needed)

The base policy for this env outputs zero actions.
"""
import math
from typing import Optional, Union

import numpy as np
import gym
from gym import spaces


class CartPoleEnv(gym.Env):
    """CartPole environment returning dict observations for the pixel-based pipeline.

    Observations:
        - 'image': (H, W, 3) uint8 rendered image
        - 'state': (4,) float32 [x, x_dot, theta, theta_dot]

    Actions:
        - Box(-1, 1, shape=(1,))  continuous force

    Reward:
        - -|theta_deg|  (minimize pole angle)

    Success:
        - |theta| < 5 degrees

    Episode ends when time_step > horizon (100 by default).
    """

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, render_size: int = 200, horizon: int = 100):
        super().__init__()
        self.horizon = horizon
        self.render_size = render_size

        # Physics
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02
        self.kinematics_integrator = "euler"

        self.theta_threshold_radians = 30 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Continuous action
        self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)

        # State bounds (for low-dim state)
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        # We expose both image and state in observation_space
        # but the actual obs is a dict returned by step/reset
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=(render_size, render_size, 3), dtype=np.uint8),
            'state': spaces.Box(low=-high, high=high, dtype=np.float32),
        })

        self.state = None
        self.time_step = 0
        self.steps_beyond_done = None

        # Pygame surfaces for offscreen rendering (lazy init)
        self._surface = None
        self._pygame_init = False

    def step(self, action):
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state

        force = self.force_mag * float(np.clip(action[0], -1.0, 1.0))
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (
            force + self.polemass_length * theta_dot ** 2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        # Clamp state
        x = np.clip(x, -self.x_threshold * 0.8, self.x_threshold * 0.8)
        theta = np.clip(theta, -self.theta_threshold_radians, self.theta_threshold_radians)

        self.state = (x, x_dot, theta, theta_dot)
        self.time_step += 1

        done = bool(self.time_step > self.horizon)

        if not done:
            reward = -abs(np.rad2deg(theta))
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = -abs(np.rad2deg(theta))
        else:
            self.steps_beyond_done += 1
            reward = 0.0

        info = {}
        info['success'] = 1 if abs(np.rad2deg(theta)) < 5 else 0

        obs = self._get_obs()
        return obs, reward, done, info

    def reset(self, *, seed: Optional[int] = None, **kwargs):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        self.time_step = 0
        return self._get_obs()

    def seed(self, seed: Optional[int] = None):
        super().reset(seed=seed)

    def _get_obs(self):
        """Return dict observation with rendered image and state."""
        return {
            'image': self.render(mode='rgb_array'),
            'state': np.array(self.state, dtype=np.float32),
        }

    def render(self, mode='rgb_array'):
        """Render the cartpole scene to an RGB image (offscreen, no display)."""
        try:
            return self._render_pygame(mode)
        except Exception:
            # Fallback: simple numpy rendering if pygame unavailable
            return self._render_numpy()

    def _render_pygame(self, mode='rgb_array'):
        """Render using pygame (offscreen)."""
        import os
        os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

        import pygame
        from pygame import gfxdraw

        screen_width = self.render_size
        screen_height = self.render_size

        if not self._pygame_init:
            pygame.init()
            self._surface = pygame.Surface((screen_width, screen_height))
            self._pygame_init = True

        self._surface.fill((255, 255, 255))

        world_width = self.x_threshold * 2
        scale = 600 / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

        x = self.state

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = screen_width / 2.0
        carty = 40

        # Draw cart
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self._surface, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self._surface, cart_coords, (0, 0, 0))

        # Draw pole
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )
        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self._surface, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self._surface, pole_coords, (202, 152, 101))

        # Draw axle
        gfxdraw.aacircle(
            self._surface, int(cartx), int(carty + axleoffset),
            int(polewidth / 2), (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self._surface, int(cartx), int(carty + axleoffset),
            int(polewidth / 2), (129, 132, 203),
        )

        # Draw ground line
        gfxdraw.hline(self._surface, 0, screen_width, carty, (0, 0, 0))

        self._surface = pygame.transform.flip(self._surface, False, True)

        image = np.transpose(
            np.array(pygame.surfarray.pixels3d(self._surface)), axes=(1, 0, 2)
        )
        # Re-flip surface back so next frame starts correctly
        self._surface = pygame.transform.flip(self._surface, False, True)

        return image.astype(np.uint8)

    def _render_numpy(self):
        """Fallback rendering using pure numpy (no pygame)."""
        H = W = self.render_size
        img = np.ones((H, W, 3), dtype=np.uint8) * 255

        if self.state is None:
            return img

        _, _, theta, _ = self.state
        cx, cy = W // 2, H // 4

        # Draw cart (black rectangle)
        cw, ch = 30, 18
        img[cy - ch // 2:cy + ch // 2, cx - cw // 2:cx + cw // 2] = 0

        # Draw pole (brown line approximation)
        pole_len = 60
        end_x = int(cx + pole_len * math.sin(theta))
        end_y = int(cy - pole_len * math.cos(theta))

        # Simple Bresenham-ish line
        num = max(abs(end_x - cx), abs(end_y - cy), 1)
        for i in range(num + 1):
            t_ = i / num
            px = int(cx + t_ * (end_x - cx))
            py = int(cy + t_ * (end_y - cy))
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    nx_, ny_ = px + dx, py + dy
                    if 0 <= nx_ < W and 0 <= ny_ < H:
                        img[ny_, nx_] = [202, 152, 101]

        return img

    def close(self):
        if self._pygame_init:
            import pygame
            pygame.quit()
            self._pygame_init = False
