
def np_state_to_jax(env, env_name: str="Pendulum-v0"):
    """ Helper that collects env state into dict for JAX `step`. """
    if env_name in ["Pendulum-v0", "CartPole-v0",
                    "MountainCar-v0", "MountainCarContinuous-v0",
                    "Acrobot-v1"]:
        state_gym_to_jax = control_np_to_jax(env, env_name)
    elif env_name in ["Catch-bsuite"]:
        state_gym_to_jax = bsuite_np_to_jax(env, env_name)
    else:
        raise ValueError(f"{env_name} is not in set of implemented"
                         " environments.")
    return state_gym_to_jax


def control_np_to_jax(env, env_name: str="Pendulum-v0"):
    """ Collects env state of classic_control into dict for JAX `step`. """
    if env_name == "Pendulum-v0":
        state_gym_to_jax = {"theta": env.state[0],
                            "theta_dot": env.state[1],
                            "time": 0,
                            "terminal": 0}
    elif env_name == "CartPole-v0":
        state_gym_to_jax = {"x": env.state[0],
                            "x_dot": env.state[1],
                            "theta": env.state[2],
                            "theta_dot": env.state[3],
                            "time": 0,
                            "terminal": 0}
    elif env_name == "MountainCar-v0":
        state_gym_to_jax = {"position": env.state[0],
                            "velocity": env.state[1],
                            "time": 0,
                            "terminal": 0}
    elif env_name == "MountainCarContinuous-v0":
        state_gym_to_jax = {"position": env.state[0],
                            "velocity": env.state[1],
                            "time": 0,
                            "terminal": 0}
    elif env_name == "Acrobot-v1":
        state_gym_to_jax = {"joint_angle1": env.state[0],
                            "joint_angle2": env.state[1],
                            "velocity_1": env.state[2],
                            "velocity_2": env.state[3],
                            "time": 0,
                            "terminal": 0}
    return state_gym_to_jax


def bsuite_np_to_jax(env, env_name: str="Catch-bsuite"):
    """ Collects env state of bsuite into dict for JAX `step`. """
    if env_name == "Catch-bsuite":
        state_gym_to_jax = {"ball_x": env._ball_x,
                            "ball_y": env._ball_y,
                            "paddle_x": env._paddle_x,
                            "paddle_y": env._paddle_y,
                            "prev_done": env._reset_next_step,
                            "time": 0,
                            "terminal": 0}
    return state_gym_to_jax