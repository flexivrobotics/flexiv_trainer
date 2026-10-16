# Copyright 2026 Flexiv Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Small-sample statistics for success rates.

Robot evaluations are small: twenty or forty trials, not thousands. At those
sizes the textbook "rate +- 1.96 sqrt(p(1-p)/n)" interval is wrong in exactly
the cases that matter -- it collapses to zero width at 0/20 and 20/20 and can
leave [0, 1] -- so the Wilson score interval is used instead.

Two policies run on the SAME start conditions are compared on the pairs where
they disagree (one succeeded, the other failed), with an exact binomial test:
pairs where both succeeded or both failed say nothing about which is better.
That is McNemar's test, and pairing is why it needs fewer trials than comparing
two independent rates.
"""

from __future__ import annotations

import math
from statistics import NormalDist

__all__ = [
    "wilson_interval",
    "paired_exact_p",
    "trials_to_detect",
    "pairs_to_detect",
]

_Z95 = NormalDist().inv_cdf(0.975)


def wilson_interval(
    successes: int, trials: int, z: float = _Z95
) -> tuple[float, float]:
    """The Wilson score interval for a success rate; (0, 1) when nothing ran."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError(f"{successes} successes in {trials} trials is not a rate")
    if trials == 0:
        return 0.0, 1.0
    p = successes / trials
    z2 = z * z
    centre = (p + z2 / (2 * trials)) / (1 + z2 / trials)
    half = (z / (1 + z2 / trials)) * math.sqrt(
        p * (1 - p) / trials + z2 / (4 * trials * trials)
    )
    return max(0.0, centre - half), min(1.0, centre + half)


def paired_exact_p(only_a: int, only_b: int) -> float:
    """Two-sided exact p-value that A and B differ, from their discordant pairs.

    `only_a` pairs where A succeeded and B failed; `only_b` the reverse. Under no
    difference each discordant pair is a fair coin.
    """
    if only_a < 0 or only_b < 0:
        raise ValueError("pair counts cannot be negative")
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def trials_to_detect(
    p_a: float, p_b: float, *, alpha: float = 0.05, power: float = 0.8
) -> int:
    """Independent trials per arm to tell success rates `p_a` and `p_b` apart."""
    if p_a == p_b:
        raise ValueError("two equal rates cannot be told apart by any number of trials")
    z_a = NormalDist().inv_cdf(1 - alpha / 2)
    z_b = NormalDist().inv_cdf(power)
    p_bar = (p_a + p_b) / 2
    root = z_a * math.sqrt(2 * p_bar * (1 - p_bar)) + z_b * math.sqrt(
        p_a * (1 - p_a) + p_b * (1 - p_b)
    )
    return math.ceil(root * root / (p_a - p_b) ** 2)


def pairs_to_detect(
    difference: float, discordance: float, *, alpha: float = 0.05, power: float = 0.8
) -> int:
    """Paired trials to detect a `difference` in success rate.

    `discordance` is the expected fraction of pairs on which the two policies
    disagree; it must be at least the difference itself.
    """
    if not (0 < abs(difference) <= discordance <= 1):
        raise ValueError(
            "need 0 < |difference| <= discordance <= 1; a difference of "
            f"{difference} cannot come from {discordance} of the pairs"
        )
    z_a = NormalDist().inv_cdf(1 - alpha / 2)
    z_b = NormalDist().inv_cdf(power)
    root = z_a * math.sqrt(discordance) + z_b * math.sqrt(discordance - difference**2)
    return math.ceil(root * root / difference**2)
