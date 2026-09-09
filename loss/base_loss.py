import torch.nn as nn
from misc.tb_wrapper import WrappedTBWriter
if 'selfocc' in WrappedTBWriter._instance_dict:
    writer = WrappedTBWriter.get_instance('selfocc')
else:
    writer = None

class BaseLoss(nn.Module):

    """ Base loss class.
    args:
        weight: weight of current loss.
        input_keys: keys for actual inputs to calculate_loss().
            Since "inputs" may contain many different fields, we use input_keys
            to distinguish them.
        loss_func: the actual loss func to calculate loss.
    """

    def __init__(
            self, 
            weight=1.0,
            input_dict={
                'input': 'input'},
            **kwargs):
        super().__init__()
        self.weight = weight
        self.input_dict = input_dict
        self.loss_func = lambda: 0
        self.writer = writer

    # def calculate_loss(self, **kwargs):
        # return self.loss_func(*[kwargs[key] for key in self.input_keys])    

    def forward(self, inputs):
        import torch
        if self.weight == 0:
            dev = 'cuda'
            for v in inputs.values():
                if isinstance(v, torch.Tensor):
                    dev = v.device
                    break
            return torch.tensor(0.0, device=dev)
        actual_inputs = {}
        for input_key, input_val in self.input_dict.items():
            actual_inputs.update({input_key: inputs[input_val]})
        if 'metas' in inputs and hasattr(self.loss_func, '__code__'):
            import inspect
            sig = inspect.signature(self.loss_func)
            if 'metas' in sig.parameters:
                actual_inputs['metas'] = inputs['metas']
        return self.weight * self.loss_func(**actual_inputs)
