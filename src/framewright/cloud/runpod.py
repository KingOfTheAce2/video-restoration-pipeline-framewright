"""RunPod cloud provider integration for video processing."""

import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .base import (
    AuthenticationError,
    CloudError,
    CloudProvider,
    GPUInfo,
    JobExecutionError,
    JobState,
    JobStatus,
    JobSubmissionError,
    ProcessingConfig,
    StorageError,
)


# GPU type mappings for RunPod
RUNPOD_GPU_TYPES = {
    "RTX_4090": {"id": "NVIDIA GeForce RTX 4090", "vram": 24},
    "RTX_3090": {"id": "NVIDIA GeForce RTX 3090", "vram": 24},
    "A100_80GB": {"id": "NVIDIA A100 80GB PCIe", "vram": 80},
    "A100_40GB": {"id": "NVIDIA A100-SXM4-40GB", "vram": 40},
    "A6000": {"id": "NVIDIA RTX A6000", "vram": 48},
    "H100": {"id": "NVIDIA H100 PCIe", "vram": 80},
    "L40": {"id": "NVIDIA L40", "vram": 48},
}

# Serverless endpoint configuration
FRAMEWRIGHT_ENDPOINT_ID = os.environ.get("FRAMEWRIGHT_RUNPOD_ENDPOINT", "")


@dataclass
class RunPodJobInfo:
    """Internal tracking for RunPod jobs."""

    job_id: str
    pod_id: Optional[str] = None
    endpoint_id: Optional[str] = None
    request_id: Optional[str] = None
    mode: str = "serverless"  # "serverless" or "pod"
    config: Optional[ProcessingConfig] = None


class RunPodProvider(CloudProvider):
    """RunPod cloud provider for GPU-accelerated video processing.

    Supports two modes:
    1. Serverless: Uses RunPod's serverless endpoints for on-demand processing
    2. Pod: Provisions dedicated GPU pods for longer jobs

    Example:
        >>> provider = RunPodProvider(api_key="your_api_key")
        >>> provider.authenticate()
        >>> config = ProcessingConfig(
        ...     input_path="s3://bucket/input.mp4",
        ...     output_path="s3://bucket/output.mp4",
        ...     scale_factor=4,
        ... )
        >>> job_id = provider.submit_job(config)
        >>> status = provider.wait_for_completion(job_id)
    """

    API_BASE_URL = "https://api.runpod.io/v2"
    GRAPHQL_URL = "https://api.runpod.io/graphql"

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        api_base_url: Optional[str] = None,
        use_serverless: bool = True,
    ):
        """Initialize RunPod provider.

        Args:
            api_key: RunPod API key. If not provided, reads from
                     RUNPOD_API_KEY environment variable.
            endpoint_id: Serverless endpoint ID for FrameWright.
                        If not provided, reads from FRAMEWRIGHT_RUNPOD_ENDPOINT.
            api_base_url: Optional custom API base URL.
            use_serverless: If True, use serverless endpoints. If False, provision pods.
        """
        api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        super().__init__(api_key=api_key, api_base_url=api_base_url or self.API_BASE_URL)
        self._endpoint_id = endpoint_id or FRAMEWRIGHT_ENDPOINT_ID
        self._use_serverless = use_serverless
        self._jobs: Dict[str, RunPodJobInfo] = {}
        self._session = None

    def _get_session(self):
        """Get or create requests session."""
        if self._session is None:
            try:
                import requests

                self._session = requests.Session()
                self._session.headers.update(
                    {
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    }
                )
            except ImportError:
                raise CloudError(
                    "requests library required for RunPod. "
                    "Install with: pip install framewright[cloud]"
                )
        return self._session

    def authenticate(self) -> bool:
        """Authenticate with RunPod API.

        Returns:
            True if authentication successful.

        Raises:
            AuthenticationError: If API key is invalid or missing.
        """
        if not self._api_key:
            raise AuthenticationError(
                "RunPod API key not provided. Set RUNPOD_API_KEY environment variable "
                "or pass api_key parameter."
            )

        try:
            session = self._get_session()
            # Verify API key by fetching user info
            response = session.post(
                self.GRAPHQL_URL,
                json={
                    "query": """
                        query {
                            myself {
                                id
                                email
                            }
                        }
                    """
                },
            )
            response.raise_for_status()
            data = response.json()

            if "errors" in data:
                raise AuthenticationError(f"RunPod auth failed: {data['errors']}")

            self._authenticated = True
            return True

        except Exception as e:
            if isinstance(e, AuthenticationError):
                raise
            raise AuthenticationError(f"RunPod authentication failed: {e}")

    def upload_video(
        self,
        local_path: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> str:
        """Upload video to RunPod network storage.

        For serverless mode, videos should be uploaded to external storage (S3, etc.)
        and referenced by URL. This method uploads to RunPod network volumes if
        using pod mode.

        Args:
            local_path: Path to local video file.
            progress_callback: Optional callback for upload progress.

        Returns:
            Remote path/URI for the uploaded file.

        Raises:
            StorageError: If upload fails.
        """
        if not local_path.exists():
            raise StorageError(f"File not found: {local_path}")

        if self._use_serverless:
            # For serverless, we expect external storage URLs
            # Return a placeholder indicating file needs to be uploaded elsewhere
            raise StorageError(
                "Serverless mode requires videos to be uploaded to external storage "
                "(S3, GCS, etc.) first. Use a CloudStorageProvider to upload."
            )

        try:
            # Pod mode: upload to network volume
            # This requires a running pod with network volume attached
            session = self._get_session()

            # Create a temporary upload endpoint on a running pod
            file_size = local_path.stat().st_size
            remote_name = f"input/{uuid.uuid4().hex[:8]}_{local_path.name}"

            # Pod mode file transfer requires SSH/SCP access to pod network volumes.
            # This is not currently supported. Use serverless mode instead.
            raise StorageError(
                "Pod mode file upload is not supported. "
                "Please use serverless mode with external storage (S3, GCS, Azure). "
                "Set use_serverless=True and upload files to your cloud storage first."
            )

        except Exception as e:
            if isinstance(e, (StorageError, NotImplementedError)):
                raise
            raise StorageError(f"Upload failed: {e}")

    def download_result(
        self,
        remote_path: str,
        local_path: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> None:
        """Download processed video from cloud storage.

        Args:
            remote_path: Remote path/URI of the processed file.
            local_path: Local path to save the file.
            progress_callback: Optional callback for download progress.

        Raises:
            StorageError: If download fails.
        """
        if self._use_serverless:
            raise StorageError(
                "Serverless mode outputs to external storage. "
                "Use the appropriate CloudStorageProvider to download."
            )

        # Pod mode file transfer requires SSH/SCP access to pod network volumes.
        # This is not currently supported. Use serverless mode instead.
        raise StorageError(
            "Pod mode file download is not supported. "
            "Please use serverless mode with external storage (S3, GCS, Azure). "
            "Results will be available at the output path specified in your job config."
        )

    def submit_job(self, config: ProcessingConfig) -> str:
        """Submit a video processing job to RunPod.

        Args:
            config: Processing configuration.

        Returns:
            Job ID for tracking.

        Raises:
            JobSubmissionError: If job submission fails.
        """
        if not self._authenticated:
            self.authenticate()

        job_id = f"fw_{uuid.uuid4().hex[:12]}"

        if self._use_serverless:
            return self._submit_serverless_job(job_id, config)
        else:
            return self._submit_pod_job(job_id, config)

    def _submit_serverless_job(self, job_id: str, config: ProcessingConfig) -> str:
        """Submit job to serverless endpoint."""
        if not self._endpoint_id:
            raise JobSubmissionError(
                "Serverless endpoint ID not configured. "
                "Set FRAMEWRIGHT_RUNPOD_ENDPOINT environment variable "
                "or deploy a FrameWright serverless endpoint on RunPod."
            )

        try:
            session = self._get_session()

            payload = {
                "input": {
                    "job_id": job_id,
                    "config": config.to_dict(),
                }
            }

            response = session.post(
                f"{self._api_base_url}/{self._endpoint_id}/run",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            request_id = data.get("id")
            if not request_id:
                raise JobSubmissionError(f"No request ID in response: {data}")

            self._jobs[job_id] = RunPodJobInfo(
                job_id=job_id,
                endpoint_id=self._endpoint_id,
                request_id=request_id,
                mode="serverless",
                config=config,
            )

            return job_id

        except Exception as e:
            if isinstance(e, JobSubmissionError):
                raise
            raise JobSubmissionError(f"Failed to submit serverless job: {e}")

    def _submit_pod_job(self, job_id: str, config: ProcessingConfig) -> str:
        """Submit job by provisioning a GPU pod."""
        try:
            session = self._get_session()

            # Get GPU type info
            gpu_info = RUNPOD_GPU_TYPES.get(config.gpu_type, RUNPOD_GPU_TYPES["RTX_4090"])

            # GraphQL mutation to create pod
            mutation = """
                mutation createPod($input: PodFindAndDeployOnDemandInput!) {
                    podFindAndDeployOnDemand(input: $input) {
                        id
                        name
                        runtime {
                            uptimeInSeconds
                        }
                    }
                }
            """

            variables = {
                "input": {
                    "cloudType": "SECURE",
                    "gpuCount": 1,
                    "volumeInGb": 50,
                    "containerDiskInGb": 20,
                    "minVcpuCount": 4,
                    "minMemoryInGb": 16,
                    "gpuTypeId": gpu_info["id"],
                    "name": f"framewright-{job_id[:8]}",
                    "imageName": "framewright/processor:latest",
                    "dockerArgs": json.dumps(
                        {
                            "JOB_ID": job_id,
                            "CONFIG": json.dumps(config.to_dict()),
                        }
                    ),
                    "ports": "8080/http",
                    "volumeMountPath": "/workspace",
                    "env": [
                        {"key": "FRAMEWRIGHT_JOB_ID", "value": job_id},
                        {"key": "FRAMEWRIGHT_CONFIG", "value": json.dumps(config.to_dict())},
                    ],
                }
            }

            response = session.post(
                self.GRAPHQL_URL,
                json={"query": mutation, "variables": variables},
            )
            response.raise_for_status()
            data = response.json()

            if "errors" in data:
                raise JobSubmissionError(f"Pod creation failed: {data['errors']}")

            pod_data = data.get("data", {}).get("podFindAndDeployOnDemand", {})
            pod_id = pod_data.get("id")

            if not pod_id:
                raise JobSubmissionError(f"No pod ID in response: {data}")

            self._jobs[job_id] = RunPodJobInfo(
                job_id=job_id,
                pod_id=pod_id,
                mode="pod",
                config=config,
            )

            return job_id

        except Exception as e:
            if isinstance(e, JobSubmissionError):
                raise
            raise JobSubmissionError(f"Failed to create pod: {e}")

    def get_job_status(self, job_id: str) -> JobStatus:
        """Get the current status of a job.

        Args:
            job_id: Job ID to check.

        Returns:
            Current job status.

        Raises:
            CloudError: If status check fails.
        """
        if job_id not in self._jobs:
            raise CloudError(f"Unknown job ID: {job_id}")

        job_info = self._jobs[job_id]

        if job_info.mode == "serverless":
            return self._get_serverless_status(job_info)
        else:
            return self._get_pod_status(job_info)

    def _get_serverless_status(self, job_info: RunPodJobInfo) -> JobStatus:
        """Get status from serverless endpoint."""
        try:
            session = self._get_session()

            response = session.get(
                f"{self._api_base_url}/{job_info.endpoint_id}/status/{job_info.request_id}"
            )
            response.raise_for_status()
            data = response.json()

            status = data.get("status", "UNKNOWN")
            output = data.get("output", {})

            state_mapping = {
                "IN_QUEUE": JobState.QUEUED,
                "IN_PROGRESS": JobState.RUNNING,
                "COMPLETED": JobState.COMPLETED,
                "FAILED": JobState.FAILED,
                "CANCELLED": JobState.CANCELLED,
                "TIMED_OUT": JobState.TIMEOUT,
            }

            job_state = state_mapping.get(status, JobState.PENDING)

            return JobStatus(
                job_id=job_info.job_id,
                state=job_state,
                progress=output.get("progress", 0.0) if isinstance(output, dict) else 0.0,
                message=output.get("message", status) if isinstance(output, dict) else status,
                output_path=output.get("output_path") if isinstance(output, dict) else None,
                error_message=data.get("error") if job_state == JobState.FAILED else None,
                metrics=output.get("metrics", {}) if isinstance(output, dict) else {},
            )

        except Exception as e:
            raise CloudError(f"Failed to get serverless status: {e}")

    def _get_pod_status(self, job_info: RunPodJobInfo) -> JobStatus:
        """Get status from pod."""
        try:
            session = self._get_session()

            query = """
                query getPod($podId: String!) {
                    pod(input: {podId: $podId}) {
                        id
                        name
                        runtime {
                            uptimeInSeconds
                        }
                        desiredStatus
                        lastStatusChange
                    }
                }
            """

            response = session.post(
                self.GRAPHQL_URL,
                json={"query": query, "variables": {"podId": job_info.pod_id}},
            )
            response.raise_for_status()
            data = response.json()

            if "errors" in data:
                raise CloudError(f"Pod status query failed: {data['errors']}")

            pod_data = data.get("data", {}).get("pod", {})
            desired_status = pod_data.get("desiredStatus", "UNKNOWN")

            state_mapping = {
                "RUNNING": JobState.RUNNING,
                "EXITED": JobState.COMPLETED,
                "TERMINATED": JobState.CANCELLED,
            }

            job_state = state_mapping.get(desired_status, JobState.PROVISIONING)

            uptime = pod_data.get("runtime", {}).get("uptimeInSeconds", 0)

            return JobStatus(
                job_id=job_info.job_id,
                state=job_state,
                progress=0.0,  # Would need to query the pod's API for actual progress
                message=f"Pod status: {desired_status}",
                metrics={"uptime_seconds": uptime},
            )

        except Exception as e:
            raise CloudError(f"Failed to get pod status: {e}")

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job.

        Args:
            job_id: Job ID to cancel.

        Returns:
            True if cancellation was successful.
        """
        if job_id not in self._jobs:
            return False

        job_info = self._jobs[job_id]

        if job_info.mode == "serverless":
            return self._cancel_serverless_job(job_info)
        else:
            return self._terminate_pod(job_info)

    def _cancel_serverless_job(self, job_info: RunPodJobInfo) -> bool:
        """Cancel a serverless job."""
        try:
            session = self._get_session()

            response = session.post(
                f"{self._api_base_url}/{job_info.endpoint_id}/cancel/{job_info.request_id}"
            )
            return response.status_code == 200

        except Exception:
            return False

    def _terminate_pod(self, job_info: RunPodJobInfo) -> bool:
        """Terminate a pod."""
        try:
            session = self._get_session()

            mutation = """
                mutation terminatePod($podId: String!) {
                    podTerminate(input: {podId: $podId})
                }
            """

            response = session.post(
                self.GRAPHQL_URL,
                json={"query": mutation, "variables": {"podId": job_info.pod_id}},
            )
            response.raise_for_status()
            data = response.json()

            return "errors" not in data

        except Exception:
            return False

    def list_available_gpus(self) -> List[GPUInfo]:
        """List available GPU instances.

        Returns:
            List of available GPU configurations with pricing.
        """
        try:
            session = self._get_session()

            query = """
                query {
                    gpuTypes {
                        id
                        displayName
                        memoryInGb
                        secureCloud
                        communityCloud
                    }
                }
            """

            response = session.post(self.GRAPHQL_URL, json={"query": query})
            response.raise_for_status()
            data = response.json()

            if "errors" in data:
                return []

            gpus = []
            for gpu in data.get("data", {}).get("gpuTypes", []):
                if gpu.get("secureCloud") or gpu.get("communityCloud"):
                    gpus.append(
                        GPUInfo(
                            gpu_type=gpu.get("displayName", "Unknown"),
                            vram_gb=gpu.get("memoryInGb", 0),
                            price_per_hour=0.0,  # Price varies, would need to query
                            availability=1,  # Simplified
                            provider_id=gpu.get("id", ""),
                        )
                    )

            return gpus

        except Exception:
            return []

    def get_job_logs(self, job_id: str, tail: int = 100) -> str:
        """Get logs from a job.

        Args:
            job_id: Job ID to get logs for.
            tail: Number of lines to return from the end.

        Returns:
            Log output as string.
        """
        if job_id not in self._jobs:
            return f"Unknown job ID: {job_id}"

        job_info = self._jobs[job_id]

        if job_info.mode == "serverless":
            # Serverless jobs may include logs in output
            try:
                status = self.get_job_status(job_id)
                return status.metrics.get("logs", "No logs available")
            except Exception as e:
                return f"Error fetching logs: {e}"

        else:
            # Pod logs would require SSH access or log streaming API
            return "Pod logs not yet implemented. SSH into pod for logs."

    def get_estimated_cost(self, config: ProcessingConfig) -> Dict[str, Any]:
        """Estimate cost for a processing job.

        Args:
            config: Processing configuration.

        Returns:
            Cost estimate with breakdown.
        """
        # RunPod pricing is dynamic, this is a rough estimate
        gpu_costs = {
            "RTX_4090": 0.44,
            "RTX_3090": 0.22,
            "A100_40GB": 0.99,
            "A100_80GB": 1.29,
            "H100": 2.49,
        }

        gpu_cost = gpu_costs.get(config.gpu_type, 0.50)
        estimated_hours = config.max_runtime_minutes / 60

        return {
            "gpu_type": config.gpu_type,
            "cost_per_hour": gpu_cost,
            "estimated_hours": estimated_hours,
            "estimated_total": gpu_cost * estimated_hours,
            "currency": "USD",
            "note": "Estimate only. Actual cost depends on runtime.",
        }
