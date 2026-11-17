import asyncio
import json
import os
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import zmq
import zmq.asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

SAMPLE_RATE = 44_100
BLOCK_SIZE = 4_096
CHANNELS = 7
STREAM_HOST = os.getenv("STREAM_HOST", "127.0.0.1")
STREAM_PORT = int(os.getenv("STREAM_PORT", "7777"))
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


@dataclass(eq=False)
class Listener:
    websocket: WebSocket
    channel: int = 0
    queue: asyncio.Queue[bytes] = field(
        default_factory=lambda: asyncio.Queue(maxsize=4)
    )


class AudioHub:
    def __init__(self) -> None:
        self.listeners: set[Listener] = set()
        self.blocks_received = 0
        self.invalid_blocks = 0
        self.last_block_at = 0.0
        self.started_at = time.monotonic()
        self._lock = asyncio.Lock()

    async def add(self, websocket: WebSocket) -> Listener:
        listener = Listener(websocket=websocket)
        async with self._lock:
            self.listeners.add(listener)
        return listener

    async def remove(self, listener: Listener) -> None:
        async with self._lock:
            self.listeners.discard(listener)

    async def publish(self, audio: np.ndarray) -> None:
        self.blocks_received += 1
        self.last_block_at = time.monotonic()

        async with self._lock:
            listeners = tuple(self.listeners)

        payloads: dict[int, bytes] = {}
        for listener in listeners:
            channel = listener.channel
            if channel not in payloads:
                payloads[channel] = np.ascontiguousarray(
                    audio[channel], dtype="<f4"
                ).tobytes()

            if listener.queue.full():
                with suppress(asyncio.QueueEmpty):
                    listener.queue.get_nowait()
            with suppress(asyncio.QueueFull):
                listener.queue.put_nowait(payloads[channel])

    def status(self) -> dict:
        age = (
            time.monotonic() - self.last_block_at
            if self.last_block_at
            else None
        )
        return {
            "source_live": age is not None and age < 2.0,
            "last_block_seconds": round(age, 3) if age is not None else None,
            "blocks_received": self.blocks_received,
            "invalid_blocks": self.invalid_blocks,
            "listeners": len(self.listeners),
            "sample_rate": SAMPLE_RATE,
            "block_size": BLOCK_SIZE,
            "channels": CHANNELS,
            "uptime_seconds": round(time.monotonic() - self.started_at),
        }


hub = AudioHub()


async def receive_processed_audio() -> None:
    context = zmq.asyncio.Context.instance()
    socket = context.socket(zmq.PULL)
    socket.setsockopt(zmq.RCVHWM, 8)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(f"tcp://{STREAM_HOST}:{STREAM_PORT}")
    print(f"[web] Recibiendo audio procesado en {STREAM_HOST}:{STREAM_PORT}")

    try:
        while True:
            try:
                message = await socket.recv_pyobj()
                audio = np.asarray(message)

                if audio.ndim == 3 and audio.shape[0] == 1:
                    audio = audio[0]
                if audio.shape != (CHANNELS, BLOCK_SIZE):
                    hub.invalid_blocks += 1
                    print(
                        f"[web] Bloque descartado: forma {audio.shape}; "
                        f"se esperaba ({CHANNELS}, {BLOCK_SIZE})"
                    )
                    continue

                await hub.publish(audio)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[web] Error recibiendo audio: {exc}")
                await asyncio.sleep(0.2)
    finally:
        socket.close()


@asynccontextmanager
async def lifespan(_: FastAPI):
    receiver = asyncio.create_task(receive_processed_audio())
    try:
        yield
    finally:
        receiver.cancel()
        with suppress(asyncio.CancelledError):
            await receiver


app = FastAPI(title="Monitor espacial Zylia", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
async def stream_status() -> dict:
    return hub.status()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", **hub.status()}


async def websocket_sender(listener: Listener) -> None:
    while True:
        payload = await listener.queue.get()
        await listener.websocket.send_bytes(payload)


async def websocket_receiver(listener: Listener) -> None:
    while True:
        raw = await listener.websocket.receive_text()
        message = json.loads(raw)
        if message.get("type") == "select":
            channel = int(message.get("channel", -1))
            if 0 <= channel < CHANNELS:
                listener.channel = channel
                while not listener.queue.empty():
                    with suppress(asyncio.QueueEmpty):
                        listener.queue.get_nowait()
                await listener.websocket.send_json(
                    {"type": "selected", "channel": channel}
                )
        elif message.get("type") == "ping":
            await listener.websocket.send_json({"type": "pong"})


@app.websocket("/ws")
async def audio_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    listener = await hub.add(websocket)
    await websocket.send_json(
        {
            "type": "config",
            "sampleRate": SAMPLE_RATE,
            "blockSize": BLOCK_SIZE,
            "channels": CHANNELS,
        }
    )

    sender = asyncio.create_task(websocket_sender(listener))
    receiver = asyncio.create_task(websocket_receiver(listener))
    try:
        done, pending = await asyncio.wait(
            {sender, receiver}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        for task in done:
            task.result()
    except (WebSocketDisconnect, RuntimeError, json.JSONDecodeError):
        pass
    finally:
        sender.cancel()
        receiver.cancel()
        await hub.remove(listener)
