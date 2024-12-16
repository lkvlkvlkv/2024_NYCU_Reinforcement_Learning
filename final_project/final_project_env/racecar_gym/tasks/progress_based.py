from .task import Task
import numpy as np


class MaximizeProgressTask(Task):
    def __init__(self, laps: int, time_limit: float, terminate_on_collision: bool,
                 delta_progress: float = 0.0, collision_reward: float = 0.0,
                 frame_reward: float = 0.0, progress_reward: float = 100.0, n_min_rays_termination=1080):
        self._time_limit = time_limit
        self._laps = laps
        self._terminate_on_collision = terminate_on_collision
        self._n_min_rays_termination = n_min_rays_termination
        self._last_stored_progress = None
        # reward params
        self._delta_progress = delta_progress
        self._progress_reward = progress_reward
        self._collision_reward = collision_reward
        self._frame_reward = frame_reward

    def reward(self, agent_id, state, action) -> float:
        agent_state = state[agent_id]
        progress = agent_state['lap'] + agent_state['progress']
        if self._last_stored_progress is None:
            self._last_stored_progress = progress
        delta = progress - self._last_stored_progress
        if delta < -0.5:
            delta = 1 + progress - self._last_stored_progress
        else:
            delta = abs(progress - self._last_stored_progress)
            if delta > .5:  # the agent is crossing the starting line in the wrong direction
                delta = (1 - progress) + self._last_stored_progress
        if self._check_collision(agent_state):
            self.collision_reward = self._collision_reward
        else:
            self.collision_reward = 0.0
        self.progress_reward = delta * self._progress_reward
        self._last_stored_progress = progress
        return self._frame_reward + self.progress_reward + self.collision_reward

    def done(self, agent_id, state) -> bool:
        agent_state = state[agent_id]
        if self._terminate_on_collision and self._check_collision(agent_state):
            return True
        return agent_state['lap'] > self._laps or self._time_limit < agent_state['time']

    def _check_collision(self, agent_state):
        safe_margin = 0.25
        collision = agent_state['wall_collision'] or len(agent_state['opponent_collisions']) > 0
        if 'observations' in agent_state and 'lidar' in agent_state['observations']:
            n_min_rays = sum(np.where(agent_state['observations']['lidar'] <= safe_margin, 1, 0))
            return n_min_rays > self._n_min_rays_termination or collision
        return collision

    def reset(self):
        self._last_stored_progress = None


class MaximizeProgressTaskCollisionInfluenceTimeLimit(MaximizeProgressTask):
    def __init__(self, laps: int, time_limit: float, terminate_on_collision: bool,
                 delta_progress: float = 0.0,
                 collision_reward: float = 0.0,
                 frame_reward: float = 0.0,
                 progress_reward: float = 100.0,
                 n_min_rays_termination=1080,
                 collision_penalty_time_reduce=40.):

        super().__init__(laps, time_limit, terminate_on_collision, delta_progress, collision_reward, frame_reward,
                         progress_reward, n_min_rays_termination)
        self.n_collision = 0
        self.collision_penalty_time_reduce = collision_penalty_time_reduce

    def done(self, agent_id, state) -> bool:
        agent_state = state[agent_id]
        if self._terminate_on_collision and self._check_collision(agent_state):
            return True
        # Collision reduce the time limit
        total_penalty = sum(agent_state['collision_penalties'])
        lap_done = agent_state['lap'] > self._laps
        time_done = (self._time_limit - total_penalty) < agent_state['time']
        return lap_done or time_done

    def reset(self):
        super(MaximizeProgressTaskCollisionInfluenceTimeLimit, self).reset()
        self.n_collision = 0


class MaximizeProgressMaskObstacleTask(MaximizeProgressTaskCollisionInfluenceTimeLimit):
    def __init__(self, laps: int, time_limit: float, terminate_on_collision: bool,
                 delta_progress=0.0,
                 collision_reward=0,
                 frame_reward=0,
                 progress_reward=100,
                 n_min_rays_termination=1080,
                 collision_penalty_time_reduce=40.):
        super().__init__(laps, time_limit, terminate_on_collision, delta_progress, collision_reward, frame_reward,
                         progress_reward, n_min_rays_termination, collision_penalty_time_reduce)

    def reward(self, agent_id, state, action) -> float:
        super().reward(agent_id, state, action)
        distance_to_obstacle = state[agent_id]['obstacle']
        if distance_to_obstacle < .3:  # max distance = 1, meaning perfectly centered in the widest point of the track
            return 0.0 + self.collision_reward + self._frame_reward
        else:
            return self.progress_reward + self.collision_reward + self._frame_reward


class MaximizeProgressRegularizeAction(MaximizeProgressTaskCollisionInfluenceTimeLimit):
    def __init__(self, laps: int, time_limit: float, terminate_on_collision: bool,
                 delta_progress=0.0,
                 collision_reward=0,
                 frame_reward=0,
                 progress_reward=100,
                 n_min_rays_termination=1080,
                 collision_penalty_time_reduce=40.,
                 action_reg=0.25):
        super().__init__(laps, time_limit, terminate_on_collision, delta_progress, collision_reward, frame_reward,
                         progress_reward, n_min_rays_termination, collision_penalty_time_reduce)
        self._action_reg = action_reg
        self._last_action = None

    def reset(self):
        super(MaximizeProgressRegularizeAction, self).reset()
        self._last_action = None

    def reward(self, agent_id, state, action) -> float:
        """ Progress-based with action regularization: penalize sharp change in control"""
        super().reward(agent_id, state, action)
        action = np.array(list(action.values()))
        if self._last_action is not None:
            self.regularization_penalty = -self._action_reg * np.linalg.norm(action - self._last_action)
        else:
            self.regularization_penalty = 0.0
        self._last_action = action
        return self.progress_reward + self.collision_reward + self._frame_reward + self.regularization_penalty


class MaximizeProgressRegularizeActionObstaclePenaltyTask(MaximizeProgressRegularizeAction):
    def __init__(self, laps: int, time_limit: float, terminate_on_collision: bool,
                 delta_progress=0.0,
                 collision_reward=0,
                 frame_reward=0,
                 progress_reward=100,
                 n_min_rays_termination=1080,
                 collision_penalty_time_reduce=40.,
                 action_reg=0.25,
                 obstacle_penalty=20.0):
        super().__init__(laps, time_limit, terminate_on_collision, delta_progress, collision_reward, frame_reward,
                         progress_reward, n_min_rays_termination, collision_penalty_time_reduce, action_reg)
        self._obstacle_penalty = obstacle_penalty

    def reward(self, agent_id, state, action) -> float:
        super().reward(agent_id, state, action)
        distance_to_obstacle = state[agent_id]['obstacle']
        self.obstacle_penalty = -self._obstacle_penalty * (1 - distance_to_obstacle)
        return self.progress_reward + self.collision_reward + self._frame_reward + self.regularization_penalty + self.obstacle_penalty

class MaximizeProgressRegularizeActionObstacleMaskTask(MaximizeProgressRegularizeAction):
    def __init__(self, laps: int, time_limit: float, terminate_on_collision: bool,
                 delta_progress=0.0,
                 collision_reward=0,
                 frame_reward=0,
                 progress_reward=100,
                 n_min_rays_termination=1080,
                 collision_penalty_time_reduce=40.,
                 action_reg=0.25):
        super().__init__(laps, time_limit, terminate_on_collision, delta_progress, collision_reward, frame_reward,
                         progress_reward, n_min_rays_termination, collision_penalty_time_reduce, action_reg)

    def reward(self, agent_id, state, action) -> float:
        super().reward(agent_id, state, action)
        distance_to_obstacle = state[agent_id]['obstacle']
        if distance_to_obstacle < .3:  # max distance = 1, meaning perfectly centered in the widest point of the track
            return 0.0 + self.collision_reward + self._frame_reward + self.regularization_penalty
        else:
            return self.progress_reward + self.collision_reward + self._frame_reward + self.regularization_penalty

class MaximizeProgressVelocityObstaclePenaltyTask(MaximizeProgressRegularizeActionObstaclePenaltyTask):
    def __init__(self, laps: int, time_limit: float, terminate_on_collision: bool,
                 delta_progress=0.0,
                 collision_reward=0,
                 frame_reward=0,
                 progress_reward=100,
                 n_min_rays_termination=1080,
                 collision_penalty_time_reduce=40.,
                 action_reg=0.25,
                 obstacle_penalty=20.0,
                 velocity_reward=20.0):
        super().__init__(laps, time_limit, terminate_on_collision, delta_progress, collision_reward, frame_reward,
                         progress_reward, n_min_rays_termination, collision_penalty_time_reduce, action_reg, obstacle_penalty)
        self._velocity_reward = velocity_reward

    def reward(self, agent_id, state, action) -> float:
        super().reward(agent_id, state, action)
        velocity = np.linalg.norm(state[agent_id]['velocity'][:2])
        self.velocity_reward = velocity * self._velocity_reward
        return self.progress_reward + self.collision_reward + self._frame_reward + self.regularization_penalty + self.obstacle_penalty + self.velocity_reward