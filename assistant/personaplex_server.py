import argparse
import asyncio
import queue
import logging
import traceback
import sys
import multiprocessing
import os
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from av import AudioFrame
from av.audio.resampler import AudioResampler
from pydantic import BaseModel

# --- Kyutai / PersonaPlex CUDA Internal Bindings ---
from moshi.local import server as pp_server_process

# ------------------------------------------------------------------------
# 1. Configuration & Model Pass-Through Arguments
# ------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Vinkona PersonaPlex WebRTC Gateway")

# Core Model Execution Flags
parser.add_argument("--hf-repo", type=str, default="nvidia/personaplex-7b-v1", help="Hugging Face Repository")
parser.add_argument("-q", type=int, default=8, help="Quantization level (8 for 24GB VRAM via bitsandbytes)")
parser.add_argument("--debug", action="store_true", default=True, help="Force maximum telemetry")

# --- The MLX Stub Block (Prevents AttributeError Crashes) ---
# Setting these to None forces the engine to auto-fetch correct default topologies
parser.add_argument("--lm-config", type=str, default=None)
parser.add_argument("--moshi-weight", type=str, default=None)
parser.add_argument("--mimi-weight", type=str, default=None)
parser.add_argument("--audio-tokenizer", type=str, default=None)
parser.add_argument("--text-tokenizer", type=str, default=None)
parser.add_argument("--tokenizer", type=str, default=None)
parser.add_argument("--voice-prompt-dir", type=str, default=None)
parser.add_argument("--voice-prompt", type=str, default=None)

# Persona Configuration
parser.add_argument("--voice", type=str, default="NATF0", help="Native Female 0 Voice Profile")
parser.add_argument("--text-prompt", type=str, default="You are Vinkona, a highly enthusiastic, chatty, and friendly AI assistant. You answer immediately.", help="System prompt dictating personality")

# Generation Limits & Samplers
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--steps", type=int, default=50000, help="Allow continuous, infinite WebRTC streaming")
parser.add_argument("--text-temp", type=float, default=0.8)
parser.add_argument("--text-topk", type=int, default=250)
parser.add_argument("--audio-temp", type=float, default=0.8)
parser.add_argument("--audio-topk", type=int, default=250)

# Bind: loopback by default — this legacy WebRTC gateway has NO auth (the live cascade
# replaced it and is the network-facing, token-authenticated path).  Pass --host explicitly
# (e.g. 0.0.0.0) only on a trusted, isolated network and at your own risk.
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=8001)

# Parse and lock the namespace
args, unknown = parser.parse_known_args()
args.quantized = args.q

# ------------------------------------------------------------------------
# 2. Telemetry & Logging Matrix
# ------------------------------------------------------------------------
log_level = logging.DEBUG if args.debug else logging.INFO
logging.basicConfig(level=log_level, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("VinkonaServer")

# Silence the overwhelming native C++ WebRTC network spam
logging.getLogger("aiortc").setLevel(logging.WARNING)
logging.getLogger("aioice").setLevel(logging.WARNING)

logger.info("=====================================================")
logger.info("🤖 VINKONA PERSONAPLEX GATEWAY INITIALIZING")
logger.info("=====================================================")

CHUNK_SIZE = 1920  # The strict 80ms audio frame limit dictated by the temporal transformer

# --- check that the MLX engine is loading

def mlx_process_wrapper(client_to_server, server_to_client, printer_q, args):
    """ Wraps the MLX engine to catch and force-print any silent crashes. """
    print("🔥 [MLX CHILD] Process spawned successfully. Initializing...", flush=True)
    try:
        # We re-initialize the logger in the child process just to be safe
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - [MLX CORE] - %(message)s")
        print("🔥 [MLX CHILD] Handing control to Kyutai/PersonaPlex Engine...", flush=True)
        
        # NOTE: library signature is server(printer_q, client_to_server, server_to_client, args)
        pp_server_process(printer_q, client_to_server, server_to_client, args)
        
    except Exception as e:
        print("\n" + "="*60, flush=True)
        print("❌ [FATAL MLX CRASH] THE AI ENGINE DIED!", flush=True)
        print("="*60, flush=True)
        traceback.print_exc()
        print("="*60 + "\n", flush=True)
        sys.exit(1)


# ------------------------------------------------------------------------
# 3. Gateway Process Management
# ------------------------------------------------------------------------
class PersonaGateway:
    def __init__(self):
        # Multiprocessing queues bridge the async WebRTC loop with the synchronous MLX engine
        self.client_to_server = multiprocessing.Queue(maxsize=30)
        self.server_to_client = multiprocessing.Queue(maxsize=30)
        self.printer_q = multiprocessing.Queue(maxsize=100)

    def start_engine(self):
        logger.info(f"🚀 Booting MLX Graph Compilation for {args.hf_repo} Q{args.q}...")
        
        args.streaming = True 
        args.turn_based = False 
        
        # --- CHANGED: Target our wrapper instead of pp_server_process directly ---
        self.process = multiprocessing.Process(
            target=mlx_process_wrapper, 
            args=(self.client_to_server, self.server_to_client, self.printer_q, args)
        )
        self.process.start()
        logger.info(f"✅ MLX Engine isolated in background process (PID: {self.process.pid}).")

gateway = PersonaGateway()
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
async def on_startup():
    gateway.start_engine()
    asyncio.create_task(print_persona_thoughts())
    asyncio.create_task(engine_watchdog())

# ------------------------------------------------------------------------
# 4. Neural Network "Thought" Logger
# ------------------------------------------------------------------------
async def print_persona_thoughts():
    """ Extracts internal text tokens from the MLX generation loop. 
        If you see text here but hear no audio, the WebRTC routing is failing. """
    while True:
        try:
            msg = await asyncio.to_thread(gateway.printer_q.get)
            if isinstance(msg, tuple) and len(msg) == 2:
                msg_type, text = msg
                # Format to easily distinguish AI thought streams in the terminal
                print(f"🧠 [AI THOUGHT]: {text}", flush=True)
        except Exception as e:
            logger.error(f"Printer queue detached: {e}")
            break
        await asyncio.sleep(0.001)

async def engine_watchdog():
    """ Constantly monitors the MLX background process to ensure it hasn't died. """
    while True:
        if gateway.process:
            if not gateway.process.is_alive():
                exit_code = gateway.process.exitcode
                logger.error(f"💀 [WATCHDOG] THE MLX PROCESS HAS DIED! (Exit Code: {exit_code})")
                # If it died, we shouldn't just keep the WebRTC server running blindly
                break
        await asyncio.sleep(5) # Check every 5 seconds

# ------------------------------------------------------------------------
# 5. WebRTC Audio Synchronization Pipeline
# ------------------------------------------------------------------------
class PersonaAudioOutputTrack(MediaStreamTrack):
    """ Binds to the Flutter Remote Stream. Polls the AI for generated audio,
        enforces 1920-sample packet sizes, and timestamps the Opus encoder. """
    kind = "audio"
    
    def __init__(self):
        super().__init__()
        self.pts = 0

    def _generate_silence_frame(self):
        """ Injects comfort noise (silence) to prevent the Opus encoder from crashing 
            when the AI is compiling its graph or 'thinking' silently. """
        pcm_data = np.zeros(CHUNK_SIZE, dtype=np.int16).tobytes()
        frame = AudioFrame(format='s16', layout='mono', samples=CHUNK_SIZE)
        frame.planes[0].update(pcm_data)
        frame.sample_rate = 24000
        frame.pts = self.pts
        self.pts += CHUNK_SIZE
        return frame

    async def recv(self):
        try:
            # Await generation from the M4 GPU
            chunk = await asyncio.wait_for(
                asyncio.to_thread(gateway.server_to_client.get), 
                timeout=0.1
            )
            
            # Validate output tensor integrity
            if isinstance(chunk, np.ndarray) and chunk.size > 0:
                if args.debug:
                    max_vol = np.max(np.abs(chunk))
                    # If Vol is 0.0, the AI is actively generating silence tokens.
                    logger.debug(f"📤 [OUTGOING] MLX Emitted {len(chunk)} samples. Peak Vol: {max_vol:.1f}")
                
                pcm_data = chunk.astype(np.int16).tobytes()
            else:
                pcm_data = np.zeros(CHUNK_SIZE, dtype=np.int16).tobytes()
        
        except (asyncio.TimeoutError, Exception):
            # The AI is currently "listening" or executing a complex reasoning trace
            return self._generate_silence_frame()

        # Strict byte-length enforcement to prevent av.buffer.Buffer ValueError
        if len(pcm_data) != CHUNK_SIZE * 2:
            pcm_data = np.zeros(CHUNK_SIZE, dtype=np.int16).tobytes()

        # Finalize and timestamp the outgoing WebRTC packet
        frame = AudioFrame(format='s16', layout='mono', samples=CHUNK_SIZE)
        frame.planes[0].update(pcm_data)
        frame.sample_rate = 24000
        frame.pts = self.pts
        self.pts += CHUNK_SIZE
        return frame

def handle_incoming_audio(track):
    """ Intercepts the 48kHz Flutter microphone stream, dynamically downsamples 
        it to the 24kHz required by Kyutai Moshi, and batches it into 1920-sample chunks. """
    resampler = AudioResampler(format='s16', layout='mono', rate=24000)
    
    async def process_track():
        logger.info("🎤 Microphone track securely bound to Python pipeline.")
        audio_buffer = []  
        
        while True:
            try:
                frame = await track.recv()
                for r_frame in resampler.resample(frame):
                    # Flatten the numpy array to prevent 2D tensor crashes in the queue
                    audio_buffer.append(r_frame.to_ndarray().flatten().astype(np.int16))
                    
                    # Accumulate four 480-sample WebRTC frames to satisfy the 1920 MLX requirement
                    if len(audio_buffer) >= 4:
                        chunk_1920 = np.concatenate(audio_buffer)
                        
                        if args.debug:
                            max_vol = np.max(np.abs(chunk_1920))
                            logger.debug(f"📥 [INCOMING] WebRTC Buffer Sent {chunk_1920.shape} | Peak Vol: {max_vol}")
                        
                        try:
                            # THE FIX: If the pipe is full, this will fail instantly instead of freezing the server
                            gateway.client_to_server.put_nowait(chunk_1920)
                        except queue.Full:
                            logger.warning("⏳ [OVERFLOW] AI is compiling or blocked! Dropping audio frame to protect WebRTC.")
                            
                        audio_buffer = []
            except Exception as e:
                logger.warning(f"Microphone stream detached: {e}")
                break
    return process_track

# ------------------------------------------------------------------------
# 6. SDP Signaling Endpoint
# ------------------------------------------------------------------------
class Offer(BaseModel): 
    sdp: str
    type: str

@app.post("/offer")
async def offer(params: Offer):
    logger.info("📡 Inbound SDP Offer detected from Flutter client...")
    pc = RTCPeerConnection()
    
    # Bind the Output (Speaker) Pipeline
    pc.addTrack(PersonaAudioOutputTrack())
    
    @pc.on("track")
    def on_track(track):
        if track.kind == "audio": 
            asyncio.create_task(handle_incoming_audio(track)())

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"🌐 ICE Connection State: {pc.connectionState.upper()}")

    # Negotiate cryptographic keys and routing constraints
    await pc.setRemoteDescription(RTCSessionDescription(sdp=params.sdp, type=params.type))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    
    logger.info("✅ SDP Answer generated. Establishing duplex stream.")
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")