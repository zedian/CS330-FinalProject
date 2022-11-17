import copy

from torch import nn


class Shareable(nn.Module):
    """Surgical parameter sharing module.

    This is the result of some *very* hacky engineering.
    Attempts to keep the compute graphs of each module identical while
    modifying the weight structures.

    If shared_params list is empty then nothing will be shared.
    """

    def __init__(self, mdl, task_keys, shared_params=[]):
        super().__init__()
        assert isinstance(task_keys, list)
        assert isinstance(shared_params, list)

        base_module_parameters = mdl.state_dict()

        k2s = {sp: '_'.join(sp.split('.')) for i, sp in enumerate(shared_params)}

        self.shared = nn.ParameterDict({
            k2s[sp]: nn.Parameter(base_module_parameters[sp].clone())
            for sp in shared_params
        })

        self.task_mdls = nn.ModuleDict({
            task_key: copy.deepcopy(mdl)
            for task_key in task_keys
        })

        for sp in shared_params:
            parameter = self.shared[k2s[sp]]
            for task_key in task_keys:
                self._override_parameter(self.task_mdls[task_key], sp, parameter)


    def _override_parameter(self, mdl, key, param):
        # TODO(gmittal): this entire method is very sus

        assert isinstance(param, nn.Parameter), 'Must be an nn.Parameter'

        comps = key.split('.')
        ref = mdl
        while len(comps) > 1:
            comp = comps.pop(0)
            if comp in ref._modules:
                ref = ref._modules[comp]
            elif comp in ref._parameters:
                ref = ref._parameters[comp]
            else:
                raise ValueError('tree traversal failed')

        parameter_name = comps.pop()
        assert len(comps) == 0
        del ref._parameters[parameter_name]
        ref._parameters[parameter_name] = param


    def forward(self, x, task_key):
        return self.task_mdls[task_key](x)