# Owner(s): ["module: inductor"]

import gc
import unittest

import torch
import torch.nn as nn
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU_AND_TRITON


@unittest.skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
class AOTInductorMemoryTest(TestCase):
    def _compile_and_drop(self, depth: int) -> None:
        model = nn.Sequential(*[nn.Linear(512, 512) for _ in range(depth)])
        model = model.half().to(GPU_TYPE).eval()
        example = (torch.randn(64, 512, dtype=torch.float16, device=GPU_TYPE),)
        with torch.no_grad():
            gm = torch.export.export(model, example).module()
            torch._inductor.aot_compile(gm, example, options={"max_autotune": False})
            del gm
        del model, example

    def test_aot_compile_releases_the_weights(self):
        """aot_compile must not keep the model's parameters alive.

        AOTI attaches parameters to the graph module rather than lifting them as
        graph inputs, and that module cycles with its Graph. Left intact the
        cycle is never reclaimed, so every compile permanently retained its
        whole weight set -- which for a large model is tens of GB, once per
        compile, under everything that ran afterwards.
        """
        # Warm up: first compile pulls in machinery whose allocations are not
        # what this test is about.
        self._compile_and_drop(depth=1)
        gc.collect()
        torch.accelerator.empty_cache()

        baseline = torch.accelerator.memory_allocated()
        self._compile_and_drop(depth=4)
        gc.collect()
        torch.accelerator.empty_cache()
        after = torch.accelerator.memory_allocated()

        leaked = after - baseline
        # 4 x 512x512 fp16 weights is ~2 MB; anything near that means the
        # parameters were retained rather than freed.
        self.assertLess(
            leaked,
            512 * 1024,
            f"aot_compile retained {leaked / 1e6:.1f} MB after the model was dropped",
        )

    def test_repeated_compiles_do_not_accumulate(self):
        """The retention scaled with the number of compiles, so check growth."""
        self._compile_and_drop(depth=1)
        gc.collect()
        torch.accelerator.empty_cache()

        baseline = torch.accelerator.memory_allocated()
        for _ in range(3):
            self._compile_and_drop(depth=4)
        gc.collect()
        torch.accelerator.empty_cache()

        self.assertLess(
            torch.accelerator.memory_allocated() - baseline,
            512 * 1024,
            "memory grew across repeated aot_compile calls",
        )


if __name__ == "__main__":
    run_tests()
