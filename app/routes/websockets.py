"""WebSocket endpoints for solar-host.

NOTE: WebSocket 2.0 Architecture
--------------------------------
In the 2.0 architecture, solar-host pushes events to solar-control via
the SolarControlClient (ws_client.py). The old per-instance WebSocket
server endpoints have been removed.

This file is kept for potential future local debugging endpoints.
"""

from fastapi import APIRouter

router = APIRouter(tags=["websockets"])


# The following endpoints have been removed in WebSocket 2.0:
# - /instances/{instance_id}/logs -> Now pushed to solar-control
# - /instances/{instance_id}/state -> Now pushed to solar-control
#
# All events are now streamed through the SolarControlClient to
# solar-control, which broadcasts them to connected webui clients.
