
def get_model(hps):
    if hps.model == 'flow':
        from .flow import Model
        model = Model(hps)
    elif hps.model == 'autoreg':
        from .autoreg import Model
        model = Model(hps)
    elif hps.model == 'tan':
        from .tan import Model
        model = Model(hps)
    else:
        raise Exception()

    return model
