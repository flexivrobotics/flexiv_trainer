#!/usr/bin/env bash
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
#
# Grant non-root access to Orbbec depth cameras. Without this rule the camera
# enumerates on the USB bus but the SDK cannot open it, so the app reports the
# camera as detected-but-unavailable.
set -euo pipefail

RULES_PATH="/etc/udev/rules.d/99-obsensor-libusb.rules"
# Orbbec 3D Technology International, Inc.
VENDOR_ID="2bc5"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "udev rules only apply to Linux; nothing to do on $(uname -s)."
    exit 0
fi

if [[ ${EUID} -ne 0 ]]; then
    echo "This script writes to ${RULES_PATH} and needs root."
    echo "Re-run it as: sudo $0"
    exit 1
fi

echo "Writing ${RULES_PATH}"
cat >"${RULES_PATH}" <<EOF
# Orbbec depth cameras (Gemini / Femto / Astra series).
#
# Two rules are needed. The first covers the raw USB node, which the SDK opens
# to enumerate devices. The second covers the V4L2 nodes (/dev/videoN) that
# carry the actual streams: those default to root:video, so without this rule
# enumeration succeeds but opening the camera fails with
# "uvc_open failed ... Return Code: -6" (LIBUSB_ERROR_ACCESS).
SUBSYSTEM=="usb", ATTR{idVendor}=="${VENDOR_ID}", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="video4linux", ATTRS{idVendor}=="${VENDOR_ID}", MODE="0666", GROUP="plugdev"
EOF

echo "Reloading udev rules"
udevadm control --reload-rules
udevadm trigger

echo
echo "Done. Replug the camera for the new permissions to take effect."
echo
if lsusb 2>/dev/null | grep -qi "${VENDOR_ID}:"; then
    echo "Detected Orbbec device(s):"
    lsusb | grep -i "${VENDOR_ID}:" | sed 's/^/  /'
else
    echo "No Orbbec device is currently connected."
fi
