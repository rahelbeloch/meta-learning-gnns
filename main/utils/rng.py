def stochastic_method(func):
    # Currently does nothing except for me being able to track which functions interaction with the RNG
    # Might be important later(?)
    def wrap(*args, **kwargs):
        result = func(*args, **kwargs)
        return result

    return wrap
