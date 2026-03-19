"""Socket.IO client for connecting to solar-control.

This module provides a persistent Socket.IO connection to solar-control,
handling:
- Registration on connect (auth via api_key)
- Event streaming (logs, instance state, health)
- Reconnection (handled by Socket.IO)
"""

import asyncio
import ssl
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

try:
    import socketio

    HAS_SOCKETIO = True
except ImportError:
    socketio = None  # type: ignore
    HAS_SOCKETIO = False


def _to_http_url(ws_url: str) -> str:
    """Convert a WebSocket or HTTP URL to an HTTP base URL for Socket.IO.

    Strips any path component since Socket.IO only needs scheme://host:port.
    Handles backward-compat ws:// URLs like ``ws://host:8015/ws/host-channel``.
    """
    from urllib.parse import urlparse

    url = ws_url.strip()
    if url.startswith("wss://"):
        url = "https://" + url[6:]
    elif url.startswith("ws://"):
        url = "http://" + url[5:]
    elif not (url.startswith("http://") or url.startswith("https://")):
        url = "http://" + url

    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.hostname}"
    if parsed.port:
        base += f":{parsed.port}"
    return base


class SolarControlClient:
    """Socket.IO client for maintaining persistent connection to solar-control.

    Identification: The host identifies itself using its API key via auth
    in the connect handshake. Solar-control looks up which registered host
    has this API key and associates the connection.
    """

    NAMESPACE = "/hosts"

    def __init__(
        self,
        control_url: str,
        api_key: str,
        host_name: str = "",
        insecure: bool = False,
    ):
        self.control_url = control_url
        self.base_url = _to_http_url(control_url)
        self.api_key = api_key
        self.host_name = host_name
        self.insecure = insecure

        # Host ID assigned by solar-control after registration_ack
        self.host_id: Optional[str] = None

        self._sio: Optional["socketio.AsyncClient"] = None
        self._connected = False
        self._pending = False
        self._running = False
        self._connection_task: Optional[Any] = None

    @property
    def is_connected(self) -> bool:
        """Check if currently connected to solar-control."""
        return self._connected and self._sio is not None and self._sio.connected

    async def start(self):
        """Start the Socket.IO client and connect to solar-control."""
        if not self.control_url:
            print("SolarControlClient: No control URL configured, skipping connection")
            return

        if not HAS_SOCKETIO:
            print(
                "SolarControlClient: python-socketio not installed, skipping connection"
            )
            return

        self._running = True
        self._main_loop = asyncio.get_running_loop()
        self._connection_task = asyncio.create_task(self._run())
        print(f"SolarControlClient: Starting connection to {self.base_url}")

    async def stop(self):
        """Stop the Socket.IO client and disconnect."""
        self._running = False

        if self._connection_task:
            self._connection_task.cancel()
            try:
                await self._connection_task
            except asyncio.CancelledError:
                pass

        if self._sio:
            try:
                await self._sio.disconnect()
            except Exception:
                pass
            self._sio = None

        self._connected = False
        print("SolarControlClient: Stopped")

    async def _run(self):
        """Create client, register handlers, connect and keep running."""
        from app.config import settings as host_settings

        ssl_verify = not self.insecure
        http_session = None

        reconnect_delay = host_settings.ws_reconnect_delay
        reconnect_max_delay = host_settings.ws_reconnect_max_delay

        if self.insecure and self.base_url.startswith("https://"):
            import aiohttp

            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            connector = aiohttp.TCPConnector(ssl=ctx)
            http_session = aiohttp.ClientSession(connector=connector)

        sio = socketio.AsyncClient(
            reconnection=True,
            reconnection_attempts=0,  # Infinite
            reconnection_delay=reconnect_delay,
            reconnection_delay_max=reconnect_max_delay,
            ssl_verify=ssl_verify,
            http_session=http_session,
        )

        sio.register_namespace(_HostNamespace(self))
        self._sio = sio  # Set before connect so _on_connect can emit

        outer_backoff = reconnect_delay
        while self._running:
            try:
                await sio.connect(
                    self.base_url,
                    namespaces=[self.NAMESPACE],
                    auth={"api_key": self.api_key, "host_name": self.host_name},
                    socketio_path="socket.io",
                )
                outer_backoff = reconnect_delay  # Reset on successful connect
                await sio.wait()
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._running:
                    print(f"SolarControlClient: Connection error: {e}")
                    await asyncio.sleep(outer_backoff)
                    outer_backoff = min(outer_backoff * 2, reconnect_max_delay)

        if sio.connected:
            await sio.disconnect()

        if http_session:
            await http_session.close()

    def _on_connect(self):
        """Called when connected to /hosts namespace."""
        self._connected = True
        print(f"SolarControlClient: Connected to {self.base_url}")
        # Emit registration with instance list
        asyncio.create_task(self._send_registration())

    def _on_disconnect(self):
        """Called when disconnected."""
        self._connected = False
        self._pending = False
        self.host_id = None

    def _on_registration_ack(self, data: dict):
        """Handle registration_ack from server.

        If we were pending, this means the admin just approved us --
        re-send registration + health so solar-control has fresh data.
        """
        was_pending = self._pending
        self._pending = False
        self.host_id = data.get("host_id")
        host_name = data.get("host_name", self.host_id)
        print(f"SolarControlClient: Registered as '{host_name}' (id: {self.host_id})")
        if was_pending:
            asyncio.create_task(self._post_approval_sync())

    def _on_pending(self, data: dict):
        """Handle pending event - host is waiting for admin approval."""
        self._pending = True
        print(
            f"SolarControlClient: Waiting for admin approval (pending_id: {data.get('pending_id', '?')})"
        )

    def _on_rejected(self, data: dict):
        """Handle rejected event - admin rejected this host."""
        self._pending = False
        reason = data.get("reason", "No reason given")
        print(f"SolarControlClient: Registration rejected: {reason}")

    async def _send_registration(self):
        """Send registration event with instance list."""
        if not self._sio or not self._sio.connected:
            return

        from app.config import config_manager

        instances = []
        for instance in config_manager.get_all_instances():
            instances.append(
                {
                    "id": instance.id,
                    "alias": instance.config.alias,
                    "status": instance.status.value,
                    "port": instance.port,
                    "supported_endpoints": instance.supported_endpoints,
                    "backend_type": getattr(
                        instance.config, "backend_type", "llamacpp"
                    ),
                }
            )

        from app.memory_monitor import detect_gpu_type

        await self._sio.emit(
            "registration",
            {
                "host_name": self.host_name,
                "instances": instances,
                "roles": config_manager.roles,
                "gpu_type": detect_gpu_type(),
            },
            namespace=self.NAMESPACE,
        )

    async def _post_approval_sync(self):
        """Re-send registration, instance states, and health after approval.

        Events sent while pending were silently dropped by solar-control,
        so we push a full snapshot now.
        """
        await self._send_registration()
        await self.send_instances_update()
        await self.send_health()

    async def _emit(self, event: str, data: dict):
        """Emit event to solar-control (no-op if disconnected)."""
        if not self._sio or not self._sio.connected:
            return
        try:
            await self._sio.emit(event, data, namespace=self.NAMESPACE)
        except Exception as e:
            print(f"SolarControlClient: Emit error: {e}")

    async def send_log(
        self,
        instance_id: str,
        seq: int,
        line: str,
        timestamp: Optional[str] = None,
        level: str = "info",
    ):
        """Send a log message to solar-control.

        Args:
            instance_id: The instance ID
            seq: Log sequence number
            line: Log line content
            timestamp: Optional timestamp string (uses current local time if not provided)
            level: Log level
        """
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        await self._emit(
            "log",
            {
                "instance_id": instance_id,
                "timestamp": ts,
                "data": {"seq": seq, "line": line, "level": level},
            },
        )

    async def send_instance_state(self, instance_id: str, state: Dict[str, Any]):
        """Send instance runtime state update to solar-control."""
        await self._emit(
            "instance_state",
            {
                "instance_id": instance_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": state,
            },
        )

    async def send_health(self, memory: Optional[Dict[str, Any]] = None):
        """Send host health/memory update to solar-control."""
        from app.memory_monitor import get_memory_info, detect_gpu_type
        from app.config import config_manager

        if memory is None:
            memory = get_memory_info()

        instances = config_manager.get_all_instances()
        running_count = sum(1 for i in instances if i.status.value == "running")

        await self._emit(
            "host_health",
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "memory": memory,
                    "gpu_type": detect_gpu_type(),
                    "instance_count": len(instances),
                    "running_instance_count": running_count,
                },
            },
        )

    async def send_instances_update(self):
        """Send full instance list to solar-control.

        Called when instances change (create, delete, start, stop).
        """
        from app.config import config_manager

        instances = []
        for instance in config_manager.get_all_instances():
            instances.append(
                {
                    "id": instance.id,
                    "alias": instance.config.alias,
                    "status": instance.status.value,
                    "port": instance.port,
                    "supported_endpoints": instance.supported_endpoints,
                    "backend_type": getattr(
                        instance.config, "backend_type", "llamacpp"
                    ),
                }
            )

        await self._emit(
            "instances_update",
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {"instances": instances},
            },
        )


# Namespace handler for /hosts - must be defined after SolarControlClient
if HAS_SOCKETIO and socketio is not None:

    class _HostNamespace(socketio.AsyncClientNamespace):
        """Socket.IO /hosts namespace handler."""

        def __init__(self, client: SolarControlClient):
            super().__init__(SolarControlClient.NAMESPACE)
            self._client = client

        def on_connect(self):
            self._client._on_connect()

        def on_disconnect(self):
            self._client._on_disconnect()

        def on_registration_ack(self, data):
            self._client._on_registration_ack(data or {})

        def on_pending(self, data):
            self._client._on_pending(data or {})

        def on_rejected(self, data):
            self._client._on_rejected(data or {})


# Global client instances (initialized in main.py)
solar_control_clients: List[SolarControlClient] = []


def get_clients() -> List[SolarControlClient]:
    """Get all solar-control clients."""
    return solar_control_clients


def get_client() -> Optional[SolarControlClient]:
    """Get the first connected solar-control client (legacy compatibility)."""
    for client in solar_control_clients:
        if client.is_connected:
            return client
    return solar_control_clients[0] if solar_control_clients else None


def init_clients(settings) -> List[SolarControlClient]:
    """Initialize solar-control client from settings.

    Uses single URL (first if multiple configured) - connect to load balancer.
    """
    global solar_control_clients

    if not settings.solar_control_url:
        print("SolarControlClient: SOLAR_CONTROL_URL not configured")
        return []

    if not settings.api_key:
        print("SolarControlClient: API_KEY not configured")
        return []

    # Single URL - take first if comma-separated
    url = settings.solar_control_url.split(",")[0].strip()
    if not url:
        print("SolarControlClient: No valid URL found")
        return []

    solar_control_clients = [
        SolarControlClient(
            control_url=url,
            api_key=settings.api_key,
            host_name=settings.host_name,
            insecure=settings.insecure,
        )
    ]
    print("SolarControlClient: Configured 1 connection")
    return solar_control_clients


async def broadcast_log(
    instance_id: str,
    seq: int,
    line: str,
    timestamp: Optional[str] = None,
    level: str = "info",
):
    """Send a log message to solar-control."""
    client = get_client()
    if client:
        await client.send_log(instance_id, seq, line, timestamp, level)


async def broadcast_instance_state(instance_id: str, state: Dict[str, Any]):
    """Send instance state update to solar-control."""
    client = get_client()
    if client:
        await client.send_instance_state(instance_id, state)


async def broadcast_health(memory: Optional[Dict[str, Any]] = None):
    """Send health update to solar-control."""
    client = get_client()
    if client:
        await client.send_health(memory)


async def broadcast_instances_update():
    """Send instance list update to solar-control."""
    client = get_client()
    if client:
        await client.send_instances_update()
