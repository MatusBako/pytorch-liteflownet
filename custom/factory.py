from configparser import SectionProxy
from json import loads
from torch import optim
from torch.optim import lr_scheduler


def build_optimizer(cfg_section: SectionProxy, net_params):
    name = cfg_section['Name']
    args = [net_params] + loads(cfg_section['Args']) if cfg_section.get('Args') is not None else []
    kwargs = loads(cfg_section['Kwargs']) if cfg_section.get('Kwargs') is not None else {}

    # dynamically get optimizer constructor
    optim_constr = getattr(optim, name)
    return optim_constr(*args, **kwargs)


def build_scheduler(cfg_section: SectionProxy, optimizer):
    name = cfg_section['Name']
    args = [optimizer] + loads(cfg_section['Args']) if cfg_section.get('Args') is not None else []
    kwargs = loads(cfg_section['Kwargs']) if cfg_section.get('Kwargs') is not None else {}

    optim_constr = getattr(lr_scheduler, name)
    return optim_constr(*args, **kwargs)
