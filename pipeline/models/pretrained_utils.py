from .PretrainedModel import timmModel, TorchPretrainedModel

def load_pretrained(
    model_name: str, 
    pretrained: bool=True, 
    global_pool: str='',
    in_chans=None, 
    **kwargs, 
):
    kwargs['pretrained'] = pretrained 
    kwargs['global_pool'] = global_pool
    kwargs['in_chans'] = in_chans
    lib = model_name.split('-')[0]
    name = '-'.join(model_name.split('-')[1:])
    if lib == 'timm':
        model = timmModel(model_name=name, **kwargs)
    elif model_name.split('-')[0] == 'torch':
        model = TorchPretrainedModel(model_name=name, **kwargs) # TODO: take Viggo's implementation
    else:
        raise NotImplementedError('ensure model name starts with "timm-" or "torch-"')
    return model