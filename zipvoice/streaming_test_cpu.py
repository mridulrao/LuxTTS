import argparse
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio

try:
    import sounddevice as sd

    SOUNDDEVICE_AVAILABLE = True
except Exception:
    SOUNDDEVICE_AVAILABLE = False

from zipvoice.luxvoice import LuxTTS


@dataclass
class ChunkMetric:
    chunk_id: int
    text: str
    emitted_at_s: float
    synth_start_s: float = 0.0
    synth_end_s: float = 0.0
    first_frame_enqueued_s: float = 0.0
    first_frame_played_s: float = 0.0
    last_frame_played_s: float = 0.0
    audio_duration_s: float = 0.0
    num_frames_sent: int = 0

    @property
    def queue_wait_before_tts_s(self) -> float:
        return self.synth_start_s - self.emitted_at_s if self.synth_start_s else 0.0

    @property
    def synth_latency_s(self) -> float:
        return self.synth_end_s - self.synth_start_s if self.synth_end_s else 0.0

    @property
    def first_audio_after_emit_s(self) -> float:
        return self.first_frame_played_s - self.emitted_at_s if self.first_frame_played_s else 0.0

    @property
    def first_audio_after_synth_start_s(self) -> float:
        return self.first_frame_played_s - self.synth_start_s if self.first_frame_played_s else 0.0

    @property
    def rtf(self) -> float:
        if self.audio_duration_s <= 0:
            return float("inf")
        return self.synth_latency_s / self.audio_duration_s


@dataclass
class PipelineMetrics:
    pipeline_start_s: float = 0.0
    session_ready_s: float = 0.0
    first_chunk_ready_s: float = 0.0
    first_audio_playout_s: float = 0.0
    last_audio_playout_s: float = 0.0
    total_audio_s: float = 0.0
    total_synth_s: float = 0.0
    total_queue_wait_s: float = 0.0
    inter_chunk_gaps_s: List[float] = field(default_factory=list)
    total_silence_inserted_s: float = 0.0
    chunk_metrics: List[ChunkMetric] = field(default_factory=list)

    @property
    def session_init_time_s(self) -> float:
        return self.session_ready_s - self.pipeline_start_s if self.session_ready_s else 0.0

    @property
    def first_chunk_ready_time_s(self) -> float:
        return self.first_chunk_ready_s - self.pipeline_start_s if self.first_chunk_ready_s else 0.0

    @property
    def first_audio_playout_time_s(self) -> float:
        return self.first_audio_playout_s - self.pipeline_start_s if self.first_audio_playout_s else 0.0

    @property
    def total_rtf(self) -> float:
        if self.total_audio_s <= 0:
            return float("inf")
        return self.total_synth_s / self.total_audio_s


class IncrementalChunker:
    def __init__(self, min_chars=35, max_chars=110):
        self.buf = ""
        self.min_chars = min_chars
        self.max_chars = max_chars

    def push(self, new_text: str) -> List[str]:
        self.buf += new_text
        out = []

        while True:
            if not self.buf.strip():
                break

            if len(self.buf) >= self.max_chars:
                idx = self._best_split(self.buf[: self.max_chars])
                out.append(self.buf[:idx].strip())
                self.buf = self.buf[idx:].lstrip()
                continue

            if len(self.buf) >= self.min_chars and re.search(r"[.!?]\s*$", self.buf):
                out.append(self.buf.strip())
                self.buf = ""
                continue

            split_idx = self._comma_split(self.buf)
            if split_idx is not None and len(self.buf) >= max(self.min_chars, 55):
                out.append(self.buf[:split_idx].strip())
                self.buf = self.buf[split_idx:].lstrip()
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


def simulated_llm_stream(
    text: str,
    mode: str = "char",
    char_delay_s: float = 0.03,
    word_delay_s: float = 0.08,
):
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


class LuxTTSWorker:
    def __init__(self, repo_id: str, model_sr: int = 48000, target_sr: int = 16000):
        self.repo_id = repo_id
        self.device = "cpu"
        self.model_sr = model_sr
        self.target_sr = target_sr

        self.lux_tts = LuxTTS(repo_id, device="cpu")
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


class AgentTransport:
    def __init__(self, sample_rate: int, frame_ms: int = 20, play_audio: bool = False):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.play_audio = play_audio and SOUNDDEVICE_AVAILABLE

        self.samples_per_frame = int(sample_rate * frame_ms / 1000)
        self.bytes_per_frame = self.samples_per_frame * 2
        self.frame_duration_s = self.samples_per_frame / sample_rate

        self.q: "queue.Queue[Optional[tuple[int, bytes]]]" = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.stream = None

        self.collected_audio: List[np.ndarray] = []
        self.frame_play_timestamps: List[float] = []
        self.chunk_first_play_s: Dict[int, float] = {}
        self.chunk_last_play_s: Dict[int, float] = {}

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

    def send_pcm_chunk(self, chunk_id: int, pcm_bytes: bytes) -> float:
        enqueue_time = 0.0
        for i in range(0, len(pcm_bytes), self.bytes_per_frame):
            frame = pcm_bytes[i : i + self.bytes_per_frame]
            if len(frame) < self.bytes_per_frame:
                frame += b"\x00" * (self.bytes_per_frame - len(frame))
            now = time.perf_counter()
            if enqueue_time == 0.0:
                enqueue_time = now
            self.q.put((chunk_id, frame))
        return enqueue_time

    def finish(self):
        self.q.put(None)
        self.thread.join()
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                return

            chunk_id, frame = item
            play_started_s = time.perf_counter()
            self.frame_play_timestamps.append(play_started_s)
            self.collected_audio.append(pcm16_bytes_to_numpy(frame))
            self.chunk_first_play_s.setdefault(chunk_id, play_started_s)

            if self.stream is not None:
                self.stream.write(frame)
            else:
                time.sleep(self.frame_duration_s)

            play_ended_s = play_started_s + self.frame_duration_s
            self.chunk_last_play_s[chunk_id] = play_ended_s


def maybe_set_cpu_threads(cpu_threads: int):
    if cpu_threads > 0:
        torch.set_num_threads(cpu_threads)
        try:
            torch.set_num_interop_threads(max(1, min(4, cpu_threads)))
        except RuntimeError:
            pass


def warmup_if_needed(worker: LuxTTSWorker, encoded_prompt, args):
    if not args.prewarm:
        return
    print("[INIT] Running CPU warmup synthesis...")
    t0 = time.perf_counter()
    _, _, dur = worker.synthesize_chunk(
        text="Hello.",
        encoded_prompt=encoded_prompt,
        num_steps=args.num_steps,
        t_shift=args.t_shift,
        guidance_scale=args.guidance_scale,
        speed=args.speed,
        return_smooth=args.return_smooth,
    )
    t1 = time.perf_counter()
    print(f"[INIT] Warmup complete in {t1 - t0:.3f}s for {dur:.3f}s audio")


def run_pipeline(args):
    maybe_set_cpu_threads(args.cpu_threads)

    print("[INIT] Using device: cpu")
    print(f"[INIT] Torch CPU threads: {torch.get_num_threads()}")

    metrics = PipelineMetrics()
    metrics.pipeline_start_s = time.perf_counter()

    worker = LuxTTSWorker(
        repo_id=args.repo_id,
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

    warmup_if_needed(worker, encoded_prompt, args)
    metrics.session_ready_s = time.perf_counter()

    chunker = IncrementalChunker(min_chars=args.min_chars, max_chars=args.max_chars)
    transport = AgentTransport(
        sample_rate=args.target_sr,
        frame_ms=args.frame_ms,
        play_audio=args.play_audio,
    )
    transport.start()

    combined_audio = np.array([], dtype=np.float32)
    combined_audio_lock = threading.Lock()
    metrics_lock = threading.Lock()

    chunk_queue: "queue.Queue[Optional[ChunkMetric]]" = queue.Queue(maxsize=args.max_pending_chunks)
    synth_errors: List[BaseException] = []

    def emit_chunk(chunk_text: str, is_flush: bool = False):
        chunk_id = len(metrics.chunk_metrics) + 1
        emit_time = time.perf_counter()
        cm = ChunkMetric(
            chunk_id=chunk_id,
            text=chunk_text,
            emitted_at_s=emit_time,
        )
        with metrics_lock:
            if metrics.first_chunk_ready_s == 0.0:
                metrics.first_chunk_ready_s = emit_time
            metrics.chunk_metrics.append(cm)
        label = "Flushed final" if is_flush else "Emitted"
        print(f"[CHUNK {chunk_id}] {label}: {repr(chunk_text)}")
        chunk_queue.put(cm)

    def synth_worker():
        nonlocal combined_audio
        try:
            while True:
                cm = chunk_queue.get()
                if cm is None:
                    return

                cm.synth_start_s = time.perf_counter()
                wav_rs, pcm_bytes, dur_s = worker.synthesize_chunk(
                    text=cm.text,
                    encoded_prompt=encoded_prompt,
                    num_steps=args.num_steps,
                    t_shift=args.t_shift,
                    guidance_scale=args.guidance_scale,
                    speed=args.speed,
                    return_smooth=args.return_smooth,
                )
                cm.synth_end_s = time.perf_counter()
                cm.audio_duration_s = dur_s

                with combined_audio_lock:
                    if combined_audio.size == 0:
                        combined_audio = wav_rs
                    else:
                        combined_audio = crossfade_append(
                            combined_audio, wav_rs, sr=args.target_sr, fade_ms=args.crossfade_ms
                        )

                cm.first_frame_enqueued_s = transport.send_pcm_chunk(cm.chunk_id, pcm_bytes)
                cm.num_frames_sent = int(np.ceil(len(pcm_bytes) / transport.bytes_per_frame))

                print(
                    f"[CHUNK {cm.chunk_id}] queue_wait={cm.queue_wait_before_tts_s:.3f}s | "
                    f"synth={cm.synth_latency_s:.3f}s | "
                    f"audio={cm.audio_duration_s:.3f}s | "
                    f"rtf={cm.rtf:.3f}"
                )
        except BaseException as exc:
            synth_errors.append(exc)
            chunk_queue.put(None)

    synth_thread = threading.Thread(target=synth_worker, daemon=True)
    synth_thread.start()

    print("\n[PIPELINE] Starting simulated LLM stream on CPU...\n")
    for piece in simulated_llm_stream(
        args.text,
        mode=args.stream_mode,
        char_delay_s=args.char_delay_s,
        word_delay_s=args.word_delay_s,
    ):
        ready_chunks = chunker.push(piece)
        for chunk in ready_chunks:
            emit_chunk(chunk)

    final_chunk = chunker.flush()
    if final_chunk:
        emit_chunk(final_chunk, is_flush=True)

    chunk_queue.put(None)
    synth_thread.join()
    if synth_errors:
        raise synth_errors[0]

    transport.finish()

    for cm in metrics.chunk_metrics:
        cm.first_frame_played_s = transport.chunk_first_play_s.get(cm.chunk_id, 0.0)
        cm.last_frame_played_s = transport.chunk_last_play_s.get(cm.chunk_id, 0.0)
        metrics.total_audio_s += cm.audio_duration_s
        metrics.total_synth_s += cm.synth_latency_s
        metrics.total_queue_wait_s += cm.queue_wait_before_tts_s

        if metrics.first_audio_playout_s == 0.0 and cm.first_frame_played_s:
            metrics.first_audio_playout_s = cm.first_frame_played_s

    if transport.frame_play_timestamps:
        metrics.last_audio_playout_s = max(transport.chunk_last_play_s.values(), default=0.0)

    prev_last_played_s = 0.0
    for cm in metrics.chunk_metrics:
        if cm.first_frame_played_s and prev_last_played_s:
            gap_s = max(0.0, cm.first_frame_played_s - prev_last_played_s)
            metrics.inter_chunk_gaps_s.append(gap_s)
            metrics.total_silence_inserted_s += gap_s
        if cm.last_frame_played_s:
            prev_last_played_s = cm.last_frame_played_s

    sf.write(args.output_wav, combined_audio, args.target_sr)

    avg_gap_s = (
        sum(metrics.inter_chunk_gaps_s) / len(metrics.inter_chunk_gaps_s)
        if metrics.inter_chunk_gaps_s
        else 0.0
    )
    avg_queue_wait_s = (
        metrics.total_queue_wait_s / len(metrics.chunk_metrics) if metrics.chunk_metrics else 0.0
    )

    print("\n================ CPU ASYNC PIPELINE METRICS ================\n")
    print("Device:                             cpu")
    print(f"Torch CPU threads:                  {torch.get_num_threads()}")
    print(f"Session init time:                  {metrics.session_init_time_s:.3f}s")
    print(f"First chunk ready time:             {metrics.first_chunk_ready_time_s:.3f}s")
    print(f"First audio playout time:           {metrics.first_audio_playout_time_s:.3f}s")
    if metrics.last_audio_playout_s:
        print(
            f"Time to final audio playout end:    "
            f"{metrics.last_audio_playout_s - metrics.pipeline_start_s:.3f}s"
        )
    print(f"Average queue wait before TTS:      {avg_queue_wait_s:.3f}s")
    print(f"Total queue wait before TTS:        {metrics.total_queue_wait_s:.3f}s")
    print(f"Average inter-chunk gap:            {avg_gap_s:.3f}s")
    print(f"Max inter-chunk gap:                {max(metrics.inter_chunk_gaps_s, default=0.0):.3f}s")
    print(f"Total silence inserted:             {metrics.total_silence_inserted_s:.3f}s")
    print(f"Total synthesized audio:            {metrics.total_audio_s:.3f}s")
    print(f"Total synth compute time:           {metrics.total_synth_s:.3f}s")
    print(f"Overall RTF:                        {metrics.total_rtf:.3f}")
    print(f"Chunks synthesized:                 {len(metrics.chunk_metrics)}")
    print(f"Saved streamed output to:           {args.output_wav}")

    print("\n------------- Per Chunk -------------")
    for idx, cm in enumerate(metrics.chunk_metrics):
        preview = cm.text.replace("\n", " ")[:70]
        gap_s = metrics.inter_chunk_gaps_s[idx - 1] if idx > 0 and idx - 1 < len(metrics.inter_chunk_gaps_s) else 0.0
        print(
            f"Chunk {cm.chunk_id:02d} | "
            f"queue_wait={cm.queue_wait_before_tts_s:.3f}s | "
            f"synth={cm.synth_latency_s:.3f}s | "
            f"audio={cm.audio_duration_s:.3f}s | "
            f"rtf={cm.rtf:.3f} | "
            f"emit->play={cm.first_audio_after_emit_s:.3f}s | "
            f"gap_before={gap_s:.3f}s | "
            f"text={repr(preview)}"
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description="CPU benchmark for LuxTTS asynchronous streaming pipeline"
    )
    parser.add_argument("--repo-id", type=str, default="YatharthS/LuxTTS")
    parser.add_argument("--prompt-audio", type=str, required=True)
    parser.add_argument("--output-wav", type=str, default="streamed_output_cpu.wav")

    parser.add_argument(
        "--text",
        type=str,
        default=(
            "Hey Harshit, how are you doing? "
            "I see you are working on interesting stuff. "
            "This is a streaming test for LuxTTS, "
            "where we mimic partial LLM output, chunk the text, "
            "synthesize each chunk, and stream PCM frames like a voice agent would."
        ),
    )

    parser.add_argument("--stream-mode", type=str, choices=["char", "word"], default="char")
    parser.add_argument("--char-delay-s", type=float, default=0.025)
    parser.add_argument("--word-delay-s", type=float, default=0.08)

    parser.add_argument("--min-chars", type=int, default=35)
    parser.add_argument("--max-chars", type=int, default=110)
    parser.add_argument(
        "--max-pending-chunks",
        type=int,
        default=8,
        help="Bounded queue depth between chunker and CPU TTS worker",
    )

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
    parser.add_argument("--prewarm", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=0, help="0 keeps PyTorch default")

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_pipeline(args)
