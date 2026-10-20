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

"""How long a test waits for work another thread does."""

# A wait ends as soon as what it waits for happens, so this only bounds how
# long a failing test takes. Shared CI runners, the Windows and macOS ones above
# all, can stall a thread for a second or more: a deadline of a few seconds
# turns that stall into a failure.
WAIT_S = 15.0
