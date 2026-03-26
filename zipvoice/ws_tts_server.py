import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torchaudio
import websockets

from zipvoice.luxvoice import LuxTTS


def maybe_set_cpu_threads(cpu_threads: int):
    if cpu_threads > 0:
        torch.set_num_threads(cpu_threads)
        try:
            torch.set_num_interop_threads(max(1, min(4, cpu_threads)))
        except RuntimeError:
            pass


def float_to_pcm16_bytes(wav):
    wav = wav.clamp(-1.0, 1.0)
    wav = (wav * 32767.0).to(torch.int16)
    return wav.cpu().numpy().tobytes()


def clean_chunk_leading_edge(
    wav: torch.Tensor,
    sample_rate: int,
    trim_leading_ms: int,
    fade_in_ms: int,
) -> torch.Tensor:
    if wav.numel() == 0:
        return wav

    trim_samples = max(0, int(sample_rate * trim_leading_ms / 1000))
    if trim_samples >= wav.numel():
        trim_samples = 0
    if trim_samples > 0:
        wav = wav[trim_samples:]

    fade_samples = max(0, int(sample_rate * fade_in_ms / 1000))
    fade_samples = min(fade_samples, wav.numel())
    if fade_samples > 1:
        wav = wav.clone()
        wav[:fade_samples] *= torch.linspace(
            0.0,
            1.0,
            fade_samples,
            device=wav.device,
            dtype=wav.dtype,
        )

    return wav


@dataclass(frozen=True)
class PromptCacheKey:
    prompt_audio: str
    duration: float
    rms: float


class LuxTTSService:
    def __init__(
        self,
        args,
        repo_id: str,
        device: str,
        target_sr: int,
        model_sr: int,
        reference_dir: Optional[str] = None,
    ):
        self.repo_id = repo_id
        self.device = device
        self.target_sr = target_sr
        self.model_sr = model_sr
        self.reference_dir = Path(reference_dir).resolve() if reference_dir else None
        self.args = args

        self.lux_tts = LuxTTS(repo_id, device=device)
        self.postprocess_device = "cuda" if device == "cuda" else "cpu"
        self.resampler = torchaudio.transforms.Resample(orig_freq=model_sr, new_freq=target_sr).to(
            self.postprocess_device
        )
        self.prompt_cache: Dict[PromptCacheKey, Any] = {}
        self.default_prompt_key: Optional[PromptCacheKey] = None
        self.synth_lock = asyncio.Lock()

    def model_info(self) -> Dict[str, Any]:
        return {
            "loaded": True,
            "repo_id": self.repo_id,
            "class_name": type(self.lux_tts).__name__,
            "device": self.device,
            "sample_rate": self.target_sr,
            "audio_format": "pcm_s16le",
            "channels": 1,
            "supports_streaming": True,
            "reference_dir": str(self.reference_dir) if self.reference_dir else None,
            "default_request": {
                "ref_duration": self.args.ref_duration,
                "rms": self.args.rms,
                "num_steps": self.args.num_steps,
                "t_shift": self.args.t_shift,
                "guidance_scale": self.args.guidance_scale,
                "speed": self.args.speed,
                "return_smooth": self.args.return_smooth,
            },
            "cpu_threads": self.args.cpu_threads if self.device == "cpu" else None,
        }

    def list_reference_files(self) -> list[str]:
        if self.reference_dir is None or not self.reference_dir.exists():
            return []
        exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
        return sorted(
            p.name for p in self.reference_dir.iterdir() if p.is_file() and p.suffix.lower() in exts
        )

    def resolve_prompt_audio(self, prompt_audio: str) -> str:
        prompt_path = Path(prompt_audio)
        if prompt_path.is_file():
            return str(prompt_path.resolve())

        if self.reference_dir is not None:
            candidate = (self.reference_dir / prompt_audio).resolve()
            if candidate.is_file():
                return str(candidate)

        raise FileNotFoundError(f"Prompt audio not found: {prompt_audio}")

    def get_encoded_prompt(self, prompt_audio: str, duration: float, rms: float):
        resolved_prompt = self.resolve_prompt_audio(prompt_audio)
        key = PromptCacheKey(prompt_audio=resolved_prompt, duration=duration, rms=rms)
        cached = self.prompt_cache.get(key)
        if cached is not None:
            return cached, True

        encoded_prompt = self.lux_tts.encode_prompt(
            resolved_prompt,
            duration=duration,
            rms=rms,
        )
        self.prompt_cache[key] = encoded_prompt
        return encoded_prompt, False

    def set_default_prompt_cache(self, prompt_audio: str, duration: float, rms: float):
        resolved_prompt = self.resolve_prompt_audio(prompt_audio)
        self.default_prompt_key = PromptCacheKey(
            prompt_audio=resolved_prompt,
            duration=duration,
            rms=rms,
        )

    def get_default_prompt_cached(self):
        if self.default_prompt_key is None:
            return None
        return self.prompt_cache.get(self.default_prompt_key)

    def synthesize(self, request: Dict[str, Any]) -> Tuple[bytes, Dict[str, Any]]:
        text = request["text"]
        prompt_audio = request["prompt_audio"]
        ref_duration = float(request.get("ref_duration", self.args.ref_duration))
        rms = float(request.get("rms", self.args.rms))
        num_steps = int(request.get("num_steps", self.args.num_steps))
        t_shift = float(request.get("t_shift", self.args.t_shift))
        guidance_scale = float(request.get("guidance_scale", self.args.guidance_scale))
        speed = float(request.get("speed", self.args.speed))
        return_smooth = bool(request.get("return_smooth", self.args.return_smooth))

        t0 = time.perf_counter()
        default_prompt = self.get_default_prompt_cached()
        if (
            default_prompt is not None
            and self.default_prompt_key is not None
            and prompt_audio == Path(self.default_prompt_key.prompt_audio).name
            and ref_duration == self.default_prompt_key.duration
            and rms == self.default_prompt_key.rms
        ):
            encoded_prompt = default_prompt
            prompt_cache_hit = True
        else:
            encoded_prompt, prompt_cache_hit = self.get_encoded_prompt(
                prompt_audio,
                duration=ref_duration,
                rms=rms,
            )
        prompt_encode_done_s = time.perf_counter()

        with torch.inference_mode():
            wav = self.lux_tts.generate_speech(
                text=text,
                encode_dict=encoded_prompt,
                num_steps=num_steps,
                t_shift=t_shift,
                guidance_scale=guidance_scale,
                speed=speed,
                return_smooth=return_smooth,
            )

        synth_done_s = time.perf_counter()
        wav = wav.detach().to(device=self.postprocess_device, dtype=torch.float32).squeeze(0)
        wav = self.resampler(wav.unsqueeze(0)).squeeze(0)
        wav = clean_chunk_leading_edge(
            wav,
            sample_rate=self.target_sr,
            trim_leading_ms=self.args.trim_leading_ms,
            fade_in_ms=self.args.fade_in_ms,
        )
        postprocess_done_s = time.perf_counter()
        pcm_bytes = float_to_pcm16_bytes(wav)
        pcm_ready_s = time.perf_counter()
        audio_duration_s = wav.numel() / self.target_sr

        metrics = {
            "audio_duration_s": audio_duration_s,
            "prompt_cache_hit": prompt_cache_hit,
            "prompt_lookup_s": prompt_encode_done_s - t0,
            "synth_compute_s": synth_done_s - prompt_encode_done_s,
            "postprocess_s": postprocess_done_s - synth_done_s,
            "pcm_encode_s": pcm_ready_s - postprocess_done_s,
            "total_request_s": pcm_ready_s - t0,
            "num_samples": int(wav.numel()),
        }
        return pcm_bytes, metrics


async def send_json(ws, payload: Dict[str, Any]):
    await ws.send(json.dumps(payload))


async def handle_socket(ws, service: LuxTTSService, args):
    async for raw_message in ws:
        if isinstance(raw_message, bytes):
            await send_json(
                ws,
                {
                    "type": "error",
                    "message": "Binary client messages are not supported. Send JSON text messages.",
                },
            )
            continue

        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError as exc:
            await send_json(ws, {"type": "error", "message": f"Invalid JSON: {exc}"})
            continue

        msg_type = message.get("type")
        request_id = message.get("request_id")

        if msg_type == "ping":
            await send_json(ws, {"type": "pong", "request_id": request_id, "status": "ok"})
            continue

        if msg_type == "model_info":
            await send_json(ws, {"type": "model_info", "request_id": request_id, **service.model_info()})
            continue

        if msg_type == "list_reference_files":
            await send_json(
                ws,
                {
                    "type": "reference_files",
                    "request_id": request_id,
                    "files": service.list_reference_files(),
                },
            )
            continue

        if msg_type != "synthesize":
            await send_json(
                ws,
                {"type": "error", "request_id": request_id, "message": f"Unknown message type: {msg_type}"},
            )
            continue

        required = {"text", "prompt_audio"}
        missing = sorted(k for k in required if not message.get(k))
        if missing:
            await send_json(
                ws,
                {
                    "type": "error",
                    "request_id": request_id,
                    "message": f"Missing required fields: {', '.join(missing)}",
                },
            )
            continue

        chunk_bytes = int(message.get("chunk_bytes", args.chunk_bytes))
        async with service.synth_lock:
            try:
                started_s = time.perf_counter()
                pcm_bytes, metrics = await asyncio.to_thread(service.synthesize, message)
                synth_ready_s = time.perf_counter()
            except Exception as exc:
                await send_json(
                    ws,
                    {
                        "type": "error",
                        "request_id": request_id,
                        "message": str(exc),
                    },
                )
                continue

            await send_json(
                ws,
                {
                    "type": "audio_start",
                    "request_id": request_id,
                    "sample_rate": service.target_sr,
                    "channels": 1,
                    "format": "pcm_s16le",
                    "chunk_bytes": chunk_bytes,
                },
            )

            first_chunk_sent_s = 0.0
            chunk_count = 0
            for i in range(0, len(pcm_bytes), chunk_bytes):
                payload = pcm_bytes[i : i + chunk_bytes]
                if not payload:
                    continue
                if first_chunk_sent_s == 0.0:
                    first_chunk_sent_s = time.perf_counter()
                chunk_count += 1
                await ws.send(payload)

            await send_json(
                ws,
                {
                    "type": "metrics",
                    "request_id": request_id,
                    "metrics": {
                        **metrics,
                        "time_to_first_audio_chunk_s": (
                            first_chunk_sent_s - started_s if first_chunk_sent_s else 0.0
                        ),
                        "server_elapsed_s": time.perf_counter() - started_s,
                        "synth_ready_s": synth_ready_s - started_s,
                        "chunks_sent": chunk_count,
                    },
                },
            )
            print(
                "[REQ]",
                json.dumps(
                    {
                        "request_id": request_id,
                        "prompt_audio": message.get("prompt_audio"),
                        "text_chars": len(message.get("text", "")),
                        "metrics": {
                            **metrics,
                            "time_to_first_audio_chunk_s": (
                                first_chunk_sent_s - started_s if first_chunk_sent_s else 0.0
                            ),
                            "server_elapsed_s": time.perf_counter() - started_s,
                            "synth_ready_s": synth_ready_s - started_s,
                            "chunks_sent": chunk_count,
                        },
                    }
                ),
                flush=True,
            )
            await send_json(
                ws,
                {
                    "type": "done",
                    "request_id": request_id,
                    "audio_bytes": len(pcm_bytes),
                    "chunks_sent": chunk_count,
                },
            )


async def async_main(args):
    if args.device == "cpu":
        maybe_set_cpu_threads(args.cpu_threads)

    service = LuxTTSService(
        args=args,
        repo_id=args.repo_id,
        device=args.device,
        target_sr=args.target_sr,
        model_sr=args.model_sr,
        reference_dir=args.reference_dir,
    )

    if args.prewarm and (args.prewarm_prompt_audio or args.prompt_audio):
        print("[INIT] Running LuxTTS prewarm synthesis...")
        try:
            service.set_default_prompt_cache(
                args.prewarm_prompt_audio or args.prompt_audio,
                duration=args.ref_duration,
                rms=args.rms,
            )
            await asyncio.to_thread(
                service.synthesize,
                {
                    "text": "Hello.",
                    "prompt_audio": args.prewarm_prompt_audio or args.prompt_audio,
                    "ref_duration": args.ref_duration,
                    "rms": args.rms,
                    "num_steps": args.num_steps,
                    "t_shift": args.t_shift,
                    "guidance_scale": args.guidance_scale,
                    "speed": args.speed,
                    "return_smooth": args.return_smooth,
                },
            )
            print("[INIT] Prewarm complete")
        except FileNotFoundError as exc:
            print(f"[INIT] Skipping prewarm: {exc}")
    elif args.prewarm:
        print("[INIT] Skipping prewarm because no default prompt audio was provided")

    print(
        f"[INIT] LuxTTS websocket server ready on ws://{args.host}:{args.port} "
        f"(device={args.device}, sample_rate={args.target_sr})"
    )
    if args.reference_dir:
        print(f"[INIT] Reference dir: {Path(args.reference_dir).resolve()}")

    async with websockets.serve(
        lambda ws: handle_socket(ws, service, args),
        args.host,
        args.port,
        max_size=None,
        ping_interval=20,
        ping_timeout=20,
    ):
        await asyncio.Future()


def build_parser():
    parser = argparse.ArgumentParser(description="Websocket LuxTTS server for LiveKit-style streaming")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--repo-id", type=str, default="YatharthS/LuxTTS")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--model-sr", type=int, default=48000)
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument(
        "--reference-dir",
        type=str,
        default=None,
        help="Optional directory of reusable prompt audio files",
    )
    parser.add_argument(
        "--chunk-bytes",
        type=int,
        default=4096,
        help="Size of each streamed PCM websocket binary frame",
    )
    parser.add_argument("--prompt-audio", type=str, default=None, help="Optional default prompt audio for prewarm")
    parser.add_argument("--prewarm-prompt-audio", type=str, default=None)
    parser.add_argument(
        "--prewarm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prewarm the model on startup when a default prompt audio is available",
    )
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--ref-duration", type=float, default=3.0)
    parser.add_argument("--rms", type=float, default=0.01)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--t-shift", type=float, default=0.5)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--speed", type=float, default=0.85)
    parser.add_argument("--return-smooth", action="store_true")
    parser.add_argument(
        "--trim-leading-ms",
        type=int,
        default=30,
        help="Trim a small leading transient from each synthesized chunk",
    )
    parser.add_argument(
        "--fade-in-ms",
        type=int,
        default=25,
        help="Apply a short fade-in to reduce click/noise at chunk start",
    )
    parser.add_argument("--stream-mode", type=str, default="word")
    parser.add_argument("--word-delay-s", type=float, default=0.12)
    parser.add_argument("--first-min-words", type=int, default=6)
    parser.add_argument("--first-target-words", type=int, default=8)
    parser.add_argument("--later-min-words", type=int, default=10)
    parser.add_argument("--later-target-words", type=int, default=14)
    parser.add_argument("--max-words", type=int, default=22)
    parser.add_argument("--max-chars", type=int, default=170)
    parser.add_argument("--lookahead-words", type=int, default=8)
    parser.add_argument("--first-chunk-max-wait-s", type=float, default=1.0)
    parser.add_argument("--later-chunk-max-wait-s", type=float, default=1.6)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    asyncio.run(async_main(args))
