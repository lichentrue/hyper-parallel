# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for the source-only external step API."""

import unittest
from collections.abc import Iterator

from hyper_parallel.distributed_data import DistributedDatasetConfig, SampleMetadata, build_distributed_dataloader
from tests.common.mark_utils import arg_mark


class _Mesh:
    mesh_shape = (1,)
    mesh_dim_names = ("dp",)
    rank_list = (0,)


class _Source:
    """Source that emits one raw-sample bin per step without checkpoint hooks."""

    def __init__(self) -> None:
        """Initialize two deterministic source steps and their cursor."""
        self.steps = [[[{"id": 0, "tokens": 4}]], [[{"id": 1, "tokens": 5}]]]
        self.position = 0

    def __iter__(self) -> Iterator[list[list[dict[str, int]]]]:
        """Yield complete local steps while advancing the source cursor."""
        while self.position < len(self.steps):
            step = self.steps[self.position]
            self.position += 1
            yield step

    def set_epoch(self, epoch: int) -> None:
        """Restart the deterministic source for any epoch.

        Args:
            epoch: Requested epoch; this fixture repeats the same samples.
        """
        del epoch
        self.position = 0


def _build(source):
    return build_distributed_dataloader(
        None,
        _Mesh(),
        DistributedDatasetConfig(seq_len=16, local_batch_size=1),
        metadata_fn=lambda sample: SampleMetadata(sample["tokens"], sample_id=sample["id"]),
        pack_fn=lambda samples, seq_len: tuple(samples) if sum(item["tokens"] for item in samples) <= seq_len else None,
        collate_fn=list,
        external_step_source=source,
        device="cpu", cost_model=lambda metadata: metadata.cost,
    )


class TestExternalStepSource(unittest.TestCase):
    """Verify source iteration, metadata, and externally owned recovery."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_source_steps_and_optional_epoch_hook(self) -> None:
        """Feature: External complete-step sources.
        Description: Consume two selected steps with the automatic balancing pipeline.
        Expectation: Membership is preserved and set_epoch restarts the external source.
        """
        source = _Source()
        loader = _build(source)
        try:
            expected = list(loader)
            self.assertEqual(expected, [[({"id": 0, "tokens": 4},)], [({"id": 1, "tokens": 5},)]])
            loader.set_epoch(1)
            self.assertEqual(list(loader), expected)
        finally:
            loader.close()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_source_does_not_require_checkpoint_hooks(self) -> None:
        """Feature: External source lifecycle validation.
        Description: Consume plain iterable steps, then wrap a caller-restored source.
        Expectation: HP consumes the supplied position without requiring checkpoint or epoch hooks.
        """
        source = [[[{"id": 0, "tokens": 4}]], [[{"id": 1, "tokens": 5}]]]
        loader = _build(source)
        try:
            self.assertEqual(list(loader), [[({"id": 0, "tokens": 4},)], [({"id": 1, "tokens": 5},)]])
        finally:
            loader.close()
        # The caller reconstructs its source at the next step before wrapping it.
        resumed = _build(source[1:])
        try:
            self.assertEqual(list(resumed), [[({"id": 1, "tokens": 5},)]])
        finally:
            resumed.close()
