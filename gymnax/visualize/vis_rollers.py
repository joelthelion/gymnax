"""Visualization for Rollers."""


def init_rollers(ax, env, state, _):
    """Initialize the visualization for Rollers.


    Args:
      ax: The matplotlib axis to draw on.
      env: The environment.
      state: The initial state.
      _: The parameters.


    Returns:
      No annotations are returned.
    """

    return ()


def update_rollers(im, _, state):
    """Update the visualization for Rollers.


    Args:
      im: The annotations (none for the moment).
      _: The environment.
      state: The current state.


    Returns:
      The updated annotations for the paddle and ball.
    """

    print(state)
    return ()
