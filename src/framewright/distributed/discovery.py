"""Node discovery for distributed render farm."""

import json
import logging
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class DiscoveryMethod(Enum):
    """Node discovery methods."""
    MULTICAST = "multicast"
    BROADCAST = "broadcast"
    STATIC = "static"
    DNS = "dns"
    CONSUL = "consul"


@dataclass
class NodeInfo:
    """Information about a render node."""
    node_id: str
    hostname: str
    address: str
    port: int

    # Capabilities
    gpu_count: int = 0
    gpu_names: List[str] = field(default_factory=list)
    gpu_memory_gb: float = 0.0
    cpu_cores: int = 1
    ram_gb: float = 8.0

    # Status
    is_available: bool = True
    current_job: Optional[str] = None
    current_chunk: Optional[str] = None

    # Performance
    estimated_fps: float = 1.0

    # Timestamps
    last_seen: datetime = field(default_factory=datetime.now)
    registered_at: datetime = field(default_factory=datetime.now)

    # Version
    version: str = "2.0.0"

    def is_stale(self, timeout_seconds: float = 60.0) -> bool:
        """Check if node info is stale (not seen recently)."""
        return (datetime.now() - self.last_seen).total_seconds() > timeout_seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "hostname": self.hostname,
            "address": self.address,
            "port": self.port,
            "gpu_count": self.gpu_count,
            "gpu_names": self.gpu_names,
            "gpu_memory_gb": self.gpu_memory_gb,
            "cpu_cores": self.cpu_cores,
            "ram_gb": self.ram_gb,
            "is_available": self.is_available,
            "current_job": self.current_job,
            "current_chunk": self.current_chunk,
            "estimated_fps": self.estimated_fps,
            "last_seen": self.last_seen.isoformat(),
            "registered_at": self.registered_at.isoformat(),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeInfo":
        """Create from dictionary."""
        info = cls(
            node_id=data["node_id"],
            hostname=data["hostname"],
            address=data["address"],
            port=data["port"],
            gpu_count=data.get("gpu_count", 0),
            gpu_names=data.get("gpu_names", []),
            gpu_memory_gb=data.get("gpu_memory_gb", 0.0),
            cpu_cores=data.get("cpu_cores", 1),
            ram_gb=data.get("ram_gb", 8.0),
            is_available=data.get("is_available", True),
            current_job=data.get("current_job"),
            current_chunk=data.get("current_chunk"),
            estimated_fps=data.get("estimated_fps", 1.0),
            version=data.get("version", "2.0.0"),
        )

        if data.get("last_seen"):
            info.last_seen = datetime.fromisoformat(data["last_seen"])
        if data.get("registered_at"):
            info.registered_at = datetime.fromisoformat(data["registered_at"])

        return info


class NodeDiscovery:
    """Handles discovery and tracking of render nodes."""

    # Multicast settings
    MULTICAST_GROUP = "239.255.42.99"
    MULTICAST_PORT = 19999
    DISCOVERY_MAGIC = b"FWRK"

    def __init__(
        self,
        method: DiscoveryMethod = DiscoveryMethod.MULTICAST,
        static_nodes: Optional[List[str]] = None,
        announce_interval: float = 10.0,
        stale_timeout: float = 60.0,
    ):
        self.method = method
        self.static_nodes = static_nodes or []
        self.announce_interval = announce_interval
        self.stale_timeout = stale_timeout

        self._nodes: Dict[str, NodeInfo] = {}
        self._lock = threading.Lock()

        self._discovery_socket: Optional[socket.socket] = None
        self._announce_socket: Optional[socket.socket] = None

        self._running = False
        self._discovery_thread: Optional[threading.Thread] = None
        self._cleanup_thread: Optional[threading.Thread] = None

        self._callbacks: List[Callable[[NodeInfo, str], None]] = []

    def add_callback(self, callback: Callable[[NodeInfo, str], None]) -> None:
        """Add callback for node events (node_info, event_type)."""
        self._callbacks.append(callback)

    def _notify(self, node: NodeInfo, event: str) -> None:
        """Notify callbacks of node event."""
        for callback in self._callbacks:
            try:
                callback(node, event)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def start(self) -> None:
        """Start node discovery."""
        if self._running:
            return

        self._running = True

        if self.method == DiscoveryMethod.MULTICAST:
            self._start_multicast_discovery()
        elif self.method == DiscoveryMethod.BROADCAST:
            self._start_broadcast_discovery()
        elif self.method == DiscoveryMethod.STATIC:
            self._load_static_nodes()

        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

        logger.info(f"Node discovery started ({self.method.value})")

    def stop(self) -> None:
        """Stop node discovery."""
        self._running = False

        if self._discovery_socket:
            try:
                self._discovery_socket.close()
            except Exception:
                pass

        if self._announce_socket:
            try:
                self._announce_socket.close()
            except Exception:
                pass

        logger.info("Node discovery stopped")

    def _start_multicast_discovery(self) -> None:
        """Start multicast-based discovery."""
        try:
            # Create socket for receiving
            self._discovery_socket = socket.socket(
                socket.AF_INET,
                socket.SOCK_DGRAM,
                socket.IPPROTO_UDP
            )
            self._discovery_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # Bind to port
            self._discovery_socket.bind(("", self.MULTICAST_PORT))

            # Join multicast group
            mreq = struct.pack(
                "4sl",
                socket.inet_aton(self.MULTICAST_GROUP),
                socket.INADDR_ANY
            )
            self._discovery_socket.setsockopt(
                socket.IPPROTO_IP,
                socket.IP_ADD_MEMBERSHIP,
                mreq
            )

            # Start listening thread
            self._discovery_thread = threading.Thread(
                target=self._multicast_listen_loop,
                daemon=True
            )
            self._discovery_thread.start()

        except Exception as e:
            logger.error(f"Failed to start multicast discovery: {e}")

    def _start_broadcast_discovery(self) -> None:
        """Start broadcast-based discovery."""
        try:
            self._discovery_socket = socket.socket(
                socket.AF_INET,
                socket.SOCK_DGRAM,
                socket.IPPROTO_UDP
            )
            self._discovery_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._discovery_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            self._discovery_socket.bind(("", self.MULTICAST_PORT))

            self._discovery_thread = threading.Thread(
                target=self._broadcast_listen_loop,
                daemon=True
            )
            self._discovery_thread.start()

        except Exception as e:
            logger.error(f"Failed to start broadcast discovery: {e}")

    def _multicast_listen_loop(self) -> None:
        """Listen for multicast announcements."""
        self._discovery_socket.settimeout(1.0)

        while self._running:
            try:
                data, addr = self._discovery_socket.recvfrom(4096)
                self._handle_discovery_message(data, addr)
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    logger.error(f"Multicast receive error: {e}")
                    time.sleep(1.0)

    def _broadcast_listen_loop(self) -> None:
        """Listen for broadcast announcements."""
        self._discovery_socket.settimeout(1.0)

        while self._running:
            try:
                data, addr = self._discovery_socket.recvfrom(4096)
                self._handle_discovery_message(data, addr)
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    logger.error(f"Broadcast receive error: {e}")
                    time.sleep(1.0)

    def _handle_discovery_message(self, data: bytes, addr: tuple) -> None:
        """Process a discovery message."""
        if not data.startswith(self.DISCOVERY_MAGIC):
            return

        try:
            json_data = data[len(self.DISCOVERY_MAGIC):].decode("utf-8")
            node_data = json.loads(json_data)

            # Update address from actual source
            node_data["address"] = addr[0]

            node = NodeInfo.from_dict(node_data)
            self._register_node(node)

        except Exception as e:
            logger.warning(f"Invalid discovery message from {addr}: {e}")

    def _register_node(self, node: NodeInfo) -> None:
        """Register or update a node."""
        with self._lock:
            is_new = node.node_id not in self._nodes
            node.last_seen = datetime.now()
            self._nodes[node.node_id] = node

        if is_new:
            logger.info(f"New node discovered: {node.hostname} ({node.address}:{node.port})")
            self._notify(node, "joined")
        else:
            self._notify(node, "updated")

    def _load_static_nodes(self) -> None:
        """Load statically configured nodes."""
        for node_str in self.static_nodes:
            try:
                # Parse "hostname:port" or "ip:port"
                parts = node_str.split(":")
                address = parts[0]
                port = int(parts[1]) if len(parts) > 1 else 8765

                # Resolve hostname
                try:
                    resolved = socket.gethostbyname(address)
                except socket.gaierror:
                    resolved = address

                node = NodeInfo(
                    node_id=f"static_{address}_{port}",
                    hostname=address,
                    address=resolved,
                    port=port,
                )
                self._register_node(node)

            except Exception as e:
                logger.warning(f"Failed to add static node {node_str}: {e}")

    def _cleanup_loop(self) -> None:
        """Periodically remove stale nodes."""
        while self._running:
            time.sleep(self.announce_interval)

            stale_nodes = []
            with self._lock:
                for node_id, node in list(self._nodes.items()):
                    if node.is_stale(self.stale_timeout):
                        stale_nodes.append(node)
                        del self._nodes[node_id]

            for node in stale_nodes:
                logger.info(f"Node stale, removing: {node.hostname}")
                self._notify(node, "left")

    def announce(self, node_info: NodeInfo) -> None:
        """Announce this node to the network."""
        if self.method not in (DiscoveryMethod.MULTICAST, DiscoveryMethod.BROADCAST):
            return

        try:
            if self._announce_socket is None:
                self._announce_socket = socket.socket(
                    socket.AF_INET,
                    socket.SOCK_DGRAM,
                    socket.IPPROTO_UDP
                )
                if self.method == DiscoveryMethod.MULTICAST:
                    self._announce_socket.setsockopt(
                        socket.IPPROTO_IP,
                        socket.IP_MULTICAST_TTL,
                        2
                    )
                else:
                    self._announce_socket.setsockopt(
                        socket.SOL_SOCKET,
                        socket.SO_BROADCAST,
                        1
                    )

            # Build message
            json_data = json.dumps(node_info.to_dict())
            message = self.DISCOVERY_MAGIC + json_data.encode("utf-8")

            if self.method == DiscoveryMethod.MULTICAST:
                self._announce_socket.sendto(
                    message,
                    (self.MULTICAST_GROUP, self.MULTICAST_PORT)
                )
            else:
                self._announce_socket.sendto(
                    message,
                    ("<broadcast>", self.MULTICAST_PORT)
                )

        except Exception as e:
            logger.warning(f"Announcement failed: {e}")

    def get_node(self, node_id: str) -> Optional[NodeInfo]:
        """Get node by ID."""
        with self._lock:
            return self._nodes.get(node_id)

    def get_all_nodes(self) -> List[NodeInfo]:
        """Get all known nodes."""
        with self._lock:
            return list(self._nodes.values())

    def get_available_nodes(self) -> List[NodeInfo]:
        """Get nodes that are available for work."""
        with self._lock:
            return [
                node for node in self._nodes.values()
                if node.is_available and not node.is_stale(self.stale_timeout)
            ]

    def get_nodes_for_job(
        self,
        min_gpu_memory: float = 0.0,
        min_ram: float = 0.0,
        max_count: int = 0
    ) -> List[NodeInfo]:
        """Get nodes suitable for a job."""
        available = self.get_available_nodes()

        # Filter by requirements
        suitable = [
            node for node in available
            if node.gpu_memory_gb >= min_gpu_memory and node.ram_gb >= min_ram
        ]

        # Sort by capability (GPU memory, then CPU cores)
        suitable.sort(
            key=lambda n: (n.gpu_memory_gb, n.cpu_cores, n.estimated_fps),
            reverse=True
        )

        if max_count > 0:
            suitable = suitable[:max_count]

        return suitable

    def update_node_status(
        self,
        node_id: str,
        is_available: Optional[bool] = None,
        current_job: Optional[str] = None,
        current_chunk: Optional[str] = None,
        estimated_fps: Optional[float] = None,
    ) -> None:
        """Update status of a node."""
        with self._lock:
            if node_id not in self._nodes:
                return

            node = self._nodes[node_id]
            if is_available is not None:
                node.is_available = is_available
            if current_job is not None:
                node.current_job = current_job if current_job else None
            if current_chunk is not None:
                node.current_chunk = current_chunk if current_chunk else None
            if estimated_fps is not None:
                node.estimated_fps = estimated_fps
            node.last_seen = datetime.now()

    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get statistics about the cluster."""
        nodes = self.get_all_nodes()
        available = self.get_available_nodes()

        return {
            "total_nodes": len(nodes),
            "available_nodes": len(available),
            "busy_nodes": len(nodes) - len(available),
            "total_gpus": sum(n.gpu_count for n in nodes),
            "available_gpus": sum(n.gpu_count for n in available),
            "total_gpu_memory_gb": sum(n.gpu_memory_gb for n in nodes),
            "total_cpu_cores": sum(n.cpu_cores for n in nodes),
            "total_ram_gb": sum(n.ram_gb for n in nodes),
            "estimated_cluster_fps": sum(n.estimated_fps for n in available),
        }
