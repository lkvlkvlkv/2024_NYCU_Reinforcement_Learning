from typing import Type
from .task import Task
from .progress_based import MaximizeProgressTask, MaximizeProgressRegularizeAction, MaximizeProgressTaskCollisionInfluenceTimeLimit, MaximizeProgressMaskObstacleTask, MaximizeProgressRegularizeActionObstaclePenaltyTask, MaximizeProgressVelocityObstaclePenaltyTask, MaximizeProgressRegularizeActionObstacleMaskTask, CircleTask
from .tracking import WaypointFollow

_registry = {}

def get_task(name: str) -> Type[Task]:
    return _registry[name]

def register_task(name: str, task: Type[Task]):
    if name not in _registry.keys():
        _registry[name] = task


register_task('maximize_progress', task=MaximizeProgressTask)
register_task('maximize_progress_collision_time_reduce', task=MaximizeProgressTaskCollisionInfluenceTimeLimit)
register_task('maximize_progress_action_reg', task=MaximizeProgressRegularizeAction)
register_task('maximize_progress_mask_obstacle', task=MaximizeProgressMaskObstacleTask)
register_task('maximize_progress_action_reg_obstacle_penalty', task=MaximizeProgressRegularizeActionObstaclePenaltyTask)
register_task('maximize_progress_action_reg_obstacle_mask', task=MaximizeProgressRegularizeActionObstacleMaskTask)
register_task('maximize_progress_velocity_obstacle_penalty', task=MaximizeProgressVelocityObstaclePenaltyTask)
register_task('circle_task', task=CircleTask)
register_task('max_tracking', task=WaypointFollow)