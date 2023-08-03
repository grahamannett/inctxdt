import unittest

import torch

from inctxdt.model_output import ModelOutput


class TestModelOutput(unittest.TestCase):
    def test_output(self):
        x = torch.rand(5, 3)

        output = ModelOutput(logits=x, only_logits=True)
        output1 = ModelOutput(x, only_logits=True)
        output2 = ModelOutput(x)
        output3 = ModelOutput(logits=x)
        assert isinstance(output, torch.Tensor)
        assert isinstance(output1, torch.Tensor)
        assert isinstance(output2.logits, torch.Tensor)
        assert isinstance(output3.logits, torch.Tensor)

    def test_with_extra(self):
        x = torch.rand(5, 3)
        extra_var = {"a": 1, "b": 2}
        output = ModelOutput(logits=x, extra=extra_var, only_logits=True)
        assert isinstance(output, torch.Tensor)

        output = ModelOutput(logits=x, extra=extra_var)
        assert isinstance(output, ModelOutput)

        for k, v in output:
            assert k in ["logits", "extra"]
