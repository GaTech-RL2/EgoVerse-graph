"""Explicit native LIBERO suite recipes; shared model/science stays unchanged."""
from dataclasses import dataclass

@dataclass(frozen=True)
class NativeLaunchProfile:
    suite: str
    name: str
    replay_environment: str
    task_uids: tuple
    @property
    def experiment(self):
        return 'libero/' + self.name

PROFILES = {
    'libero10': NativeLaunchProfile('libero10', 'action_flow_libero10_h240_euler50_dithalf_80k_s42', 'LIBERO10_REPLAY_ROOT', tuple(range(30, 40))),
    'libero_object': NativeLaunchProfile('libero_object', 'action_flow_libero_object_h240_euler50_dithalf_80k_s42', 'LIBERO_OBJECT_REPLAY_ROOT', tuple(range(10, 20))),
    'libero_goal': NativeLaunchProfile('libero_goal', 'action_flow_libero_goal_h240_euler50_dithalf_80k_s42', 'LIBERO_GOAL_REPLAY_ROOT', tuple(range(20, 30))),
    'libero_spatial': NativeLaunchProfile('libero_spatial', 'action_flow_libero_spatial_h240_euler50_dithalf_80k_s42', 'LIBERO_SPATIAL_REPLAY_ROOT', tuple(range(10))),
}

def profile_for_suite(suite):
    try: return PROFILES[suite]
    except (KeyError, TypeError): raise ValueError('unsupported native LIBERO suite: ' + str(suite)) from None

def profile_for_experiment(experiment):
    for profile in PROFILES.values():
        if profile.experiment == experiment: return profile
    raise ValueError('unsupported native LIBERO experiment: ' + str(experiment))

def profile_for_config(config):
    profile = profile_for_suite(config['benchmark']['suite'])
    if config.get('name') != profile.name:
        raise ValueError('native suite/recipe identity mismatch')
    for mode in ('train', 'valid'):
        if config['data'][mode + '_datasets']['libero_panda']['resolver']['suite'] != profile.suite:
            raise ValueError('native suite/resolver identity mismatch')
    return profile

def profile_for_argv(argv):
    selected = [arg.split('=', 1)[1] for arg in argv if arg.split('=', 1)[0].lstrip('+') == 'experiment' and '=' in arg]
    if len(selected) != 1:
        raise ValueError('exactly one native experiment override required')
    return profile_for_experiment(selected[0])
