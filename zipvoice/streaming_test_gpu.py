import argparse
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio

# -----------------------------
# Optional live playback
# -----------------------------
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except Exception:
    SOUNDDEVICE_AVAILABLE = False

# -----------------------------
# LuxTTS import
# -----------------------------
from zipvoice.luxvoice import LuxTTS


# ============================================================
# Config / Metrics
# ============================================================

@dataclass
class ChunkMetric:
    chunk_id: int
    text: str
    emitted_at_s: float
    synth_start_s: float = 0.0
    synth_end_s: float = 0.0
    first_frame_sent_s: float = 0.0
    audio_duration_s: float = 0.0
    num_frames_sent: int = 0

    @property
    def synth_latency_s(self) -> float:
        return self.synth_end_s - self.synth_start_s

    @property
    def wait_from_emit_to_synth_start_s(self) -> float:
        return self.synth_start_s - self.emitted_at_s

    @property
    def first_audio_after_chunk_emit_s(self) -> float:
        if self.first_frame_sent_s == 0.0:
            return 0.0
        return self.first_frame_sent_s - self.emitted_at_s

    @property
    def rtf(self) -> float:
        if self.audio_duration_s <= 0:
            return float("inf")
        return self.synth_latency_s / self.audio_duration_s


@dataclass
class PipelineMetrics:
    pipeline_start_s: float = 0.0
    llm_start_s: float = 0.0
    first_text_seen_s: float = 0.0
    first_chunk_emitted_s: float = 0.0
    first_pcm_sent_s: float = 0.0
    last_pcm_sent_s: float = 0.0
    total_audio_s: float = 0.0
    total_synth_s: float = 0.0
    chunk_metrics: List[ChunkMetric] = field(default_factory=list)

    @property
    def ttfb_s(self) -> float:
        if self.first_pcm_sent_s == 0.0:
            return 0.0
        return self.first_pcm_sent_s - self.pipeline_start_s

    @property
    def time_to_first_chunk_s(self) -> float:
        if self.first_chunk_emitted_s == 0.0:
            return 0.0
        return self.first_chunk_emitted_s - self.pipeline_start_s

    @property
    def total_rtf(self) -> float:
        if self.total_audio_s <= 0:
            return float("inf")
        return self.total_synth_s / self.total_audio_s


# ============================================================
# Incremental chunker
# ============================================================

class IncrementalChunker:
    """
    A practical streaming text chunker.
    Emits chunk when:
      - sentence-ending punctuation appears and enough text exists
      - buffer exceeds max_chars
      - comma/semicolon split if enough text exists
    """
    def __init__(self, min_chars=35, max_chars=120):
        self.buf = ""
        self.min_chars = min_chars
        self.max_chars = max_chars

    def push(self, new_text: str) -> List[str]:
        self.buf += new_text
        out = []

        while True:
            stripped = self.buf.strip()
            if not stripped:
                break

            if len(self.buf) >= self.max_chars:
                idx = self._best_split(self.buf[:self.max_chars])
                out.append(self.buf[:idx].strip())
                self.buf = self.buf[idx:].lstrip()
                continue

            if len(self.buf) >= self.min_chars and re.search(r"[.!?]\s*$", self.buf):
                out.append(self.buf.strip())
                self.buf = ""
                continue

            comma_idx = self._comma_split(self.buf)
            if len(self.buf) >= max(self.min_chars, 55) and comma_idx is not None:
                out.append(self.buf[:comma_idx].strip())
                self.buf = self.buf[comma_idx:].lstrip()
                continue

            break

        return [x for x in out if x]

    def flush(self) -> Optional[str]:
        text = self.buf.strip()
        self.buf = ""
        return text if text else None

    def _best_split(self, text: str) -> int:
        candidates = []
        for pat in [". ", "! ", "? ", "; ", ": ", ", "]:
            idx = text.rfind(pat)
            if idx != -1:
                candidates.append(idx + 1)
        if candidates:
            return max(candidates)

        space_idx = text.rfind(" ")
        return space_idx if space_idx != -1 else len(text)

    def _comma_split(self, text: str) -> Optional[int]:
        matches = list(re.finditer(r"[,;:]\s+", text))
        if not matches:
            return None
        return matches[-1].end()


# ============================================================
# Simulated LLM streaming
# ============================================================

def simulated_llm_stream(
    text: str,
    mode: str = "char",
    char_delay_s: float = 0.03,
    word_delay_s: float = 0.08,
):
    """
    Yields partial text like a streaming LLM.
    mode = char | word
    """
    if mode == "char":
        for ch in text:
            yield ch
            time.sleep(char_delay_s)
    elif mode == "word":
        parts = re.findall(r"\S+\s*|\n", text)
        for p in parts:
            yield p
            time.sleep(word_delay_s)
    else:
        raise ValueError(f"Unknown stream mode: {mode}")


# ============================================================
# PCM helpers
# ============================================================

def float_to_pcm16_bytes(wav: np.ndarray) -> bytes:
    wav = np.clip(wav, -1.0, 1.0)
    return (wav * 32767.0).astype(np.int16).tobytes()


def pcm16_bytes_to_numpy(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767.0


def crossfade_append(prev: np.ndarray, nxt: np.ndarray, sr: int, fade_ms: int = 40) -> np.ndarray:
    if prev.size == 0:
        return nxt
    fade_samples = int(sr * fade_ms / 1000)
    fade_samples = min(fade_samples, len(prev), len(nxt))
    if fade_samples <= 0:
        return np.concatenate([prev, nxt])

    a = prev[-fade_samples:]
    b = nxt[:fade_samples]
    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    mixed = a * fade_out + b * fade_in
    return np.concatenate([prev[:-fade_samples], mixed, nxt[fade_samples:]])


# ============================================================
# LuxTTS worker
# ============================================================

class LuxTTSWorker:
    def __init__(
        self,
        repo_id: str,
        device: str,
        model_sr: int = 48000,
        target_sr: int = 16000,
    ):
        self.repo_id = repo_id
        self.device = device
        self.model_sr = model_sr
        self.target_sr = target_sr

        self.lux_tts = LuxTTS(repo_id, device=device)
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=model_sr,
            new_freq=target_sr,
        )

    def encode_prompt(self, prompt_audio: str, duration: float, rms: float):
        return self.lux_tts.encode_prompt(prompt_audio, duration=duration, rms=rms)

    def synthesize_chunk(
        self,
        text: str,
        encoded_prompt,
        num_steps: int,
        t_shift: float,
        guidance_scale: float,
        speed: float,
        return_smooth: bool,
    ):
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

        wav = wav.detach().cpu().numpy().squeeze().astype(np.float32)
        wav_t = torch.from_numpy(wav).unsqueeze(0)
        wav_rs = self.resampler(wav_t).squeeze(0).cpu().numpy().astype(np.float32)
        pcm = float_to_pcm16_bytes(wav_rs)
        duration_s = len(wav_rs) / self.target_sr
        return wav_rs, pcm, duration_s


# ============================================================
# Transport
# ============================================================

class AgentTransport:
    """
    Simulates agent/telephony transport consuming PCM frames.
    Can optionally play audio in real time if sounddevice is installed.
    """
    def __init__(self, sample_rate: int, frame_ms: int = 20, play_audio: bool = False):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.play_audio = play_audio and SOUNDDEVICE_AVAILABLE

        self.samples_per_frame = int(sample_rate * frame_ms / 1000)
        self.bytes_per_frame = self.samples_per_frame * 2  # int16 mono

        self.q = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

        self.stream = None
        self.collected_audio: List[np.ndarray] = []
        self.frame_send_timestamps: List[float] = []

    def start(self):
        if self.play_audio:
            self.stream = sd.RawOutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=self.samples_per_frame,
            )
            self.stream.start()
        self.thread.start()

    def send_pcm_chunk(self, pcm_bytes: bytes):
        for i in range(0, len(pcm_bytes), self.bytes_per_frame):
            frame = pcm_bytes[i:i + self.bytes_per_frame]
            if len(frame) < self.bytes_per_frame:
                # pad final frame
                frame += b"\x00" * (self.bytes_per_frame - len(frame))
            self.q.put(frame)

    def finish(self):
        self.q.put(None)
        self.thread.join()
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()

    def _run(self):
        while not self.stop_event.is_set():
            item = self.q.get()
            if item is None:
                return
            self.frame_send_timestamps.append(time.perf_counter())
            self.collected_audio.append(pcm16_bytes_to_numpy(item))
            if self.stream is not None:
                self.stream.write(item)


# ============================================================
# Pipeline runner
# ============================================================

def run_pipeline(args):
    metrics = PipelineMetrics()
    metrics.pipeline_start_s = time.perf_counter()
    metrics.llm_start_s = metrics.pipeline_start_s

    worker = LuxTTSWorker(
        repo_id=args.repo_id,
        device=args.device,
        model_sr=48000,
        target_sr=args.target_sr,
    )

    print("[INIT] Encoding prompt...")
    t0 = time.perf_counter()
    encoded_prompt = worker.encode_prompt(
        prompt_audio=args.prompt_audio,
        duration=args.ref_duration,
        rms=args.rms,
    )
    t1 = time.perf_counter()
    print(f"[INIT] Prompt encoded in {t1 - t0:.3f}s")

    chunker = IncrementalChunker(min_chars=args.min_chars, max_chars=args.max_chars)
    transport = AgentTransport(
        sample_rate=args.target_sr,
        frame_ms=args.frame_ms,
        play_audio=args.play_audio,
    )
    transport.start()

    combined_audio = np.array([], dtype=np.float32)
    chunk_id = 0

    first_text_seen = False

    print("\n[PIPELINE] Starting simulated LLM stream...\n")
    for piece in simulated_llm_stream(
        args.text,
        mode=args.stream_mode,
        char_delay_s=args.char_delay_s,
        word_delay_s=args.word_delay_s,
    ):
        now = time.perf_counter()
        if not first_text_seen:
            metrics.first_text_seen_s = now
            first_text_seen = True

        ready_chunks = chunker.push(piece)
        for chunk in ready_chunks:
            chunk_id += 1
            emit_time = time.perf_counter()

            if metrics.first_chunk_emitted_s == 0.0:
                metrics.first_chunk_emitted_s = emit_time

            cm = ChunkMetric(
                chunk_id=chunk_id,
                text=chunk,
                emitted_at_s=emit_time,
            )

            print(f"[CHUNK {chunk_id}] Emitted: {repr(chunk)}")

            cm.synth_start_s = time.perf_counter()
            wav_rs, pcm_bytes, dur_s = worker.synthesize_chunk(
                text=chunk,
                encoded_prompt=encoded_prompt,
                num_steps=args.num_steps,
                t_shift=args.t_shift,
                guidance_scale=args.guidance_scale,
                speed=args.speed,
                return_smooth=args.return_smooth,
            )
            cm.synth_end_s = time.perf_counter()
            cm.audio_duration_s = dur_s

            if combined_audio.size == 0:
                combined_audio = wav_rs
            else:
                combined_audio = crossfade_append(
                    combined_audio, wav_rs, sr=args.target_sr, fade_ms=args.crossfade_ms
                )

            before_frames = len(transport.frame_send_timestamps)
            transport.send_pcm_chunk(pcm_bytes)

            # wait until at least one frame from this chunk is observed
            while len(transport.frame_send_timestamps) == before_frames:
                time.sleep(0.001)

            cm.first_frame_sent_s = transport.frame_send_timestamps[before_frames]
            cm.num_frames_sent = int(np.ceil(len(pcm_bytes) / transport.bytes_per_frame))

            if metrics.first_pcm_sent_s == 0.0:
                metrics.first_pcm_sent_s = cm.first_frame_sent_s

            metrics.total_audio_s += dur_s
            metrics.total_synth_s += cm.synth_latency_s
            metrics.chunk_metrics.append(cm)

            print(
                f"[CHUNK {chunk_id}] synth={cm.synth_latency_s:.3f}s | "
                f"audio={cm.audio_duration_s:.3f}s | "
                f"rtf={cm.rtf:.3f} | "
                f"first_audio_after_emit={cm.first_audio_after_chunk_emit_s:.3f}s"
            )

    final_chunk = chunker.flush()
    if final_chunk:
        chunk_id += 1
        emit_time = time.perf_counter()
        if metrics.first_chunk_emitted_s == 0.0:
            metrics.first_chunk_emitted_s = emit_time

        cm = ChunkMetric(
            chunk_id=chunk_id,
            text=final_chunk,
            emitted_at_s=emit_time,
        )

        print(f"[CHUNK {chunk_id}] Flushed final: {repr(final_chunk)}")

        cm.synth_start_s = time.perf_counter()
        wav_rs, pcm_bytes, dur_s = worker.synthesize_chunk(
            text=final_chunk,
            encoded_prompt=encoded_prompt,
            num_steps=args.num_steps,
            t_shift=args.t_shift,
            guidance_scale=args.guidance_scale,
            speed=args.speed,
            return_smooth=args.return_smooth,
        )
        cm.synth_end_s = time.perf_counter()
        cm.audio_duration_s = dur_s

        if combined_audio.size == 0:
            combined_audio = wav_rs
        else:
            combined_audio = crossfade_append(
                combined_audio, wav_rs, sr=args.target_sr, fade_ms=args.crossfade_ms
            )

        before_frames = len(transport.frame_send_timestamps)
        transport.send_pcm_chunk(pcm_bytes)

        while len(transport.frame_send_timestamps) == before_frames:
            time.sleep(0.001)

        cm.first_frame_sent_s = transport.frame_send_timestamps[before_frames]
        cm.num_frames_sent = int(np.ceil(len(pcm_bytes) / transport.bytes_per_frame))

        if metrics.first_pcm_sent_s == 0.0:
            metrics.first_pcm_sent_s = cm.first_frame_sent_s

        metrics.total_audio_s += dur_s
        metrics.total_synth_s += cm.synth_latency_s
        metrics.chunk_metrics.append(cm)

        print(
            f"[CHUNK {chunk_id}] synth={cm.synth_latency_s:.3f}s | "
            f"audio={cm.audio_duration_s:.3f}s | "
            f"rtf={cm.rtf:.3f} | "
            f"first_audio_after_emit={cm.first_audio_after_chunk_emit_s:.3f}s"
        )

    transport.finish()
    if transport.frame_send_timestamps:
        metrics.last_pcm_sent_s = transport.frame_send_timestamps[-1]

    # Save final reconstructed stream
    sf.write(args.output_wav, combined_audio, args.target_sr)

    print("\n================ METRICS ================\n")
    print(f"Pipeline start:             {metrics.pipeline_start_s:.6f}")
    print(f"First chunk emitted after:  {metrics.time_to_first_chunk_s:.3f}s")
    print(f"TTFB (first PCM frame):     {metrics.ttfb_s:.3f}s")
    if metrics.last_pcm_sent_s:
        print(f"Time to last PCM frame:     {metrics.last_pcm_sent_s - metrics.pipeline_start_s:.3f}s")
    print(f"Total synthesized audio:    {metrics.total_audio_s:.3f}s")
    print(f"Total synth compute time:   {metrics.total_synth_s:.3f}s")
    print(f"Overall RTF:                {metrics.total_rtf:.3f}")
    print(f"Chunks synthesized:         {len(metrics.chunk_metrics)}")
    print(f"Saved streamed output to:   {args.output_wav}")

    print("\n------------- Per Chunk -------------")
    for cm in metrics.chunk_metrics:
        preview = cm.text.replace("\n", " ")[:70]
        print(
            f"Chunk {cm.chunk_id:02d} | "
            f"synth={cm.synth_latency_s:.3f}s | "
            f"audio={cm.audio_duration_s:.3f}s | "
            f"rtf={cm.rtf:.3f} | "
            f"emit->first_audio={cm.first_audio_after_chunk_emit_s:.3f}s | "
            f"text={repr(preview)}"
        )


# ============================================================
# CLI
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--repo-id", type=str, default="YatharthS/LuxTTS")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--prompt-audio", type=str, required=True)
    parser.add_argument("--output-wav", type=str, default="streamed_output.wav")

    parser.add_argument(
        "--text",
        type=str,
        default=(
            "Hey Harshit, how are you doing? "
            "I see you are working on interesting stuff. "
            "This is a streaming test for LuxTTS, where we mimic partial LLM output, "
            "chunk the text, synthesize each chunk, and stream PCM frames like a voice agent would."
        ),
    )

    parser.add_argument("--stream-mode", type=str, choices=["char", "word"], default="char")
    parser.add_argument("--char-delay-s", type=float, default=0.025)
    parser.add_argument("--word-delay-s", type=float, default=0.08)

    parser.add_argument("--min-chars", type=int, default=35)
    parser.add_argument("--max-chars", type=int, default=110)

    parser.add_argument("--num-steps", type=int, default=3)
    parser.add_argument("--t-shift", type=float, default=0.65)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--speed", type=float, default=0.8)
    parser.add_argument("--return-smooth", action="store_true")

    parser.add_argument("--ref-duration", type=float, default=4.0)
    parser.add_argument("--rms", type=float, default=0.01)

    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--frame-ms", type=int, default=20)
    parser.add_argument("--crossfade-ms", type=int, default=40)

    parser.add_argument("--play-audio", action="store_true")

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_pipeline(args)