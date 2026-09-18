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

"""Flexiv Trainer backend package."""

import os

# PyArrow 25.0.0's bundled mimalloc can segfault when first loaded by a
# short-lived worker (recording/preview), then used by another (e.g. merge).
# Select the system allocator before any dependency imports Arrow. Keeping this
# here covers the server, CLI tools, and their child processes without eagerly
# importing PyArrow. Explicit operator configuration still takes precedence.
# https://github.com/apache/arrow/issues/50471
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

__all__ = ["__version__"]

__version__ = "0.5.0"
