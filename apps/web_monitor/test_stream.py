"""Emite una señal sintética de siete canales para probar la web sin GPU."""
import pickle
import socket
import struct
import time

import numpy as np
import zmq

context = zmq.Context()
sender = context.socket(zmq.PUSH)
sender.connect("tcp://127.0.0.1:7777")
phase = np.zeros(7)
freqs = np.array([180, 240, 320, 440, 550, 660, 880], dtype=np.float64)

try:
    while True:
        index = np.arange(4096)
        audio = np.stack([
            0.12 * np.sin(phase[channel] + 2 * np.pi * freqs[channel] * index / 44100)
            for channel in range(7)
        ]).astype(np.float32)
        phase = (phase + 2 * np.pi * freqs * 4096 / 44100) % (2 * np.pi)
        sender.send_pyobj(audio)
        time.sleep(4096 / 44100)
except KeyboardInterrupt:
    pass
finally:
    sender.close()
    context.term()
