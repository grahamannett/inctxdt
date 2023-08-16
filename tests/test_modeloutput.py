import unittest

import torch

from inctxdt.models.model_output import ModelOutput


class TestModelOutput(unittest.TestCase):
    def test_output(self):
        x = torch.rand(5, 1)
        x1 = torch.rand(5, 2)
        x2 = torch.rand(5, 3)
        x3 = torch.rand(5, 4)

        output0 = ModelOutput(logits=x, only_logits=False)
        assert isinstance(output0, ModelOutput)
        assert isinstance(output0.logits, torch.Tensor)

        output = ModelOutput(logits=x, only_logits=True)
        output1 = ModelOutput(x1, only_logits=True)
        output2 = ModelOutput(x2)
        output3 = ModelOutput(logits=x3)


        assert isinstance(output, torch.Tensor)
        assert isinstance(output1, torch.Tensor)
        assert isinstance(output2.logits, torch.Tensor)
        assert isinstance(output3.logits, torch.Tensor)
        assert isinstance(output2["logits"], torch.Tensor)

    @unittest.skip("ll")
    def test_with_extra(self):
        x = torch.rand(5, 3)
        extra_var = {"a": 1, "b": 2}
        output = ModelOutput(logits=x, extra=extra_var, only_logits=True)
        assert isinstance(output, torch.Tensor)

        output = ModelOutput(logits=x, extra=extra_var)
        assert isinstance(output, ModelOutput)

        for k, v in output:
            assert k in ["logits", "extra"]
