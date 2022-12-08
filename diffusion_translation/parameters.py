class Parameters(dict):
    '''Most of the code in the Diffusion-LM Paper just passes the args through all the functions
    I wanted to use dicts for the params, but only noticed then that they don't support getting their
    values like attributes.'''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def __getattribute__(self, __name: 'str'):
        if hasattr(super(), __name):
            return super().__getattribute__(__name) 
        elif super().get(__name):
            return super().get(__name)
        else:
            raise AttributeError(f"'parameters' object has no attribute '{__name}'")