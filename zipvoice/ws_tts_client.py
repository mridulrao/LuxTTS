import argparse
import asyncio
import json
import os
import uuid
import wave
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import websockets


@dataclass
class SynthesisResult:
    audio_bytes: bytes
    sample_rate: int
    channels: int
    audio_format: str
    metrics: Dict[str, Any]


class LuxTTSWebSocketClient:
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.websocket = None

    async def __aenter__(self):
        self.websocket = await websockets.connect(self.server_url, max_size=None)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.websocket is not None:
            await self.websocket.close()

    async def _send_json(self, payload: Dict[str, Any]):
        assert self.websocket is not None
        await self.websocket.send(json.dumps(payload))

    async def _recv_json(self) -> Dict[str, Any]:
        assert self.websocket is not None
        message = await self.websocket.recv()
        if isinstance(message, bytes):
            raise RuntimeError("Expected JSON text message but received binary audio payload")
        return json.loads(message)

    async def ping(self) -> bool:
        request_id = str(uuid.uuid4())
        await self._send_json({"type": "ping", "request_id": request_id})
        response = await self._recv_json()
        return response.get("type") == "pong" and response.get("request_id") == request_id

    async def model_info(self) -> Dict[str, Any]:
        request_id = str(uuid.uuid4())
        await self._send_json({"type": "model_info", "request_id": request_id})
        response = await self._recv_json()
        if response.get("type") == "error":
            raise RuntimeError(response.get("message", "Unknown server error"))
        return response

    async def list_reference_files(self) -> List[str]:
        request_id = str(uuid.uuid4())
        await self._send_json({"type": "list_reference_files", "request_id": request_id})
        response = await self._recv_json()
        if response.get("type") == "error":
            raise RuntimeError(response.get("message", "Unknown server error"))
        return response.get("files", [])

    async def synthesize(self, **request: Any) -> SynthesisResult:
        request_id = str(uuid.uuid4())
        payload = {"type": "synthesize", "request_id": request_id, **request}
        await self._send_json(payload)

        start = await self._recv_json()
        if start.get("type") == "error":
            raise RuntimeError(start.get("message", "Unknown server error"))
        if start.get("type") != "audio_start":
            raise RuntimeError(f"Unexpected first response: {start}")

        audio_frames = bytearray()
        metrics: Dict[str, Any] = {}

        while True:
            assert self.websocket is not None
            message = await self.websocket.recv()
            if isinstance(message, bytes):
                audio_frames.extend(message)
                continue

            payload = json.loads(message)
            if payload.get("type") == "error":
                raise RuntimeError(payload.get("message", "Unknown server error"))

            if payload.get("request_id") != request_id:
                continue

            if payload.get("type") == "metrics":
                metrics = payload.get("metrics", {})
                continue

            if payload.get("type") == "done":
                return SynthesisResult(
                    audio_bytes=bytes(audio_frames),
                    sample_rate=int(start["sample_rate"]),
                    channels=int(start["channels"]),
                    audio_format=str(start["format"]),
                    metrics=metrics,
                )

    @staticmethod
    def write_wav(path: str, audio_bytes: bytes, sample_rate: int, channels: int):
        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_bytes)


async def check_server_connection(server_url: str) -> bool:
    print(f"\nChecking websocket connection to {server_url}...")
    try:
        async with LuxTTSWebSocketClient(server_url) as client:
            ok = await client.ping()
        if ok:
            print("   Server is reachable")
            return True
        print("   Ping failed")
        return False
    except Exception as exc:
        print(f"   Connection failed: {exc}")
        return False


async def check_model_info(server_url: str) -> Optional[dict]:
    print("\nFetching model information...")
    try:
        async with LuxTTSWebSocketClient(server_url) as client:
            info = await client.model_info()
        print("   Model information retrieved:")
        print(f"      loaded: {info.get('loaded', False)}")
        print(f"      repo_id: {info.get('repo_id', 'unknown')}")
        print(f"      class_name: {info.get('class_name', 'unknown')}")
        print(f"      device: {info.get('device', 'unknown')}")
        print(f"      sample_rate: {info.get('sample_rate', 'unknown')} Hz")
        print(f"      audio_format: {info.get('audio_format', 'unknown')}")
        return info
    except Exception as exc:
        print(f"   Error: {exc}")
        return None


async def list_reference_audio(server_url: str) -> List[str]:
    print("\nFetching available reference audio files...")
    try:
        async with LuxTTSWebSocketClient(server_url) as client:
            files = await client.list_reference_files()
        if files:
            print(f"   Found {len(files)} reference file(s):")
            for i, filename in enumerate(files, start=1):
                print(f"      {i}. {filename}")
        else:
            print("   No reference files found on the server")
        return files
    except Exception as exc:
        print(f"   Error: {exc}")
        return []


async def test_synthesis(
    server_url: str,
    prompt_audio: Optional[str],
    output_wav: str,
    text: str,
    num_steps: int,
    t_shift: float,
    guidance_scale: float,
    speed: float,
    return_smooth: bool,
) -> bool:
    print("\nTesting websocket synthesis...")
    try:
        async with LuxTTSWebSocketClient(server_url) as client:
            reference_files = []
            if not prompt_audio:
                reference_files = await client.list_reference_files()
                if not reference_files:
                    print("   No prompt audio provided and no server reference files available")
                    return False
                prompt_audio = reference_files[0]
                print(f"   Using reference prompt: {prompt_audio}")

            result = await client.synthesize(
                text=text,
                prompt_audio=prompt_audio,
                num_steps=num_steps,
                t_shift=t_shift,
                guidance_scale=guidance_scale,
                speed=speed,
                return_smooth=return_smooth,
            )
            client.write_wav(output_wav, result.audio_bytes, result.sample_rate, result.channels)

        print("   Synthesis succeeded")
        print(f"      Saved wav: {output_wav}")
        for key, value in result.metrics.items():
            if isinstance(value, float):
                print(f"      {key}: {value:.3f}")
            else:
                print(f"      {key}: {value}")
        return True
    except Exception as exc:
        print(f"   Error: {exc}")
        return False


async def async_main(args):
    print("=" * 70)
    print("LUXTTS WEBSOCKET DIAGNOSTIC TOOL")
    print("=" * 70)
    print(f"\nTarget server: {args.server_url}")

    is_connected = await check_server_connection(args.server_url)
    if not is_connected:
        print("\nDiagnostic failed: server not reachable")
        return 1

    model_info = await check_model_info(args.server_url)
    reference_files = await list_reference_audio(args.server_url)
    synthesis_ok = await test_synthesis(
        args.server_url,
        prompt_audio=args.prompt_audio,
        output_wav=args.output_wav,
        text=args.text,
        num_steps=args.num_steps,
        t_shift=args.t_shift,
        guidance_scale=args.guidance_scale,
        speed=args.speed,
        return_smooth=args.return_smooth,
    )

    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"\nServer connected: {'yes' if is_connected else 'no'}")
    print(f"Model loaded: {model_info.get('loaded') if model_info else 'unknown'}")
    print(f"Reference files: {len(reference_files)}")
    print(f"Synthesis test: {'working' if synthesis_ok else 'failed'}")
    print(f"Output wav: {args.output_wav}")
    print("\nUse this client as the transport layer for a LiveKit custom TTS node.")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description="Diagnostic websocket client for LuxTTS streaming")
    parser.add_argument(
        "--server-url",
        type=str,
        default=os.getenv("LUXTTS_WS_URL", "ws://localhost:8765"),
    )
    parser.add_argument("--prompt-audio", type=str, default=None)
    parser.add_argument("--output-wav", type=str, default="ws_client_test.wav")
    parser.add_argument(
        "--text",
        type=str,
        default="This is a websocket synthesis test for LuxTTS.",
    )
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument("--t-shift", type=float, default=0.65)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--speed", type=float, default=0.8)
    parser.add_argument("--return-smooth", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(asyncio.run(async_main(build_parser().parse_args())))
