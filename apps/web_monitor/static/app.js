const CHANNELS = Array.from({ length: 7 }, (_, index) => ({
  index,
  angle: index * 50,
  label: `Posición ${index * 50}°`,
}));

const elements = {
  buttons: document.querySelector("#channelButtons"),
  angle: document.querySelector("#angleReadout"),
  number: document.querySelector("#channelNumber"),
  position: document.querySelector("#channelPosition"),
  line: document.querySelector("#directionLine").closest("svg"),
  listen: document.querySelector("#listenButton"),
  listenLabel: document.querySelector("#listenLabel"),
  notice: document.querySelector("#notice"),
  connectionDot: document.querySelector("#connectionDot"),
  connectionText: document.querySelector("#connectionText"),
  sourceStatus: document.querySelector("#sourceStatus"),
  listenerCount: document.querySelector("#listenerCount"),
  signalState: document.querySelector("#signalState"),
  waveform: document.querySelector("#waveform"),
};

let selectedChannel = 0;
let socket;
let reconnectTimer;
let audioContext;
let workletNode;
let resampler;
let listening = false;
let sourceRate = 44100;

class StreamingResampler {
  constructor(inputRate, outputRate) {
    this.ratio = inputRate / outputRate;
    this.position = 0;
    this.previous = null;
  }

  reset() {
    this.position = 0;
    this.previous = null;
  }

  process(input) {
    if (!input.length) return new Float32Array();
    const data = new Float32Array(input.length + (this.previous === null ? 0 : 1));
    let offset = 0;
    if (this.previous !== null) {
      data[0] = this.previous;
      offset = 1;
    }
    data.set(input, offset);

    const outputLength = Math.max(0, Math.ceil((data.length - 1 - this.position) / this.ratio));
    const output = new Float32Array(outputLength);
    let written = 0;
    while (this.position < data.length - 1) {
      const left = Math.floor(this.position);
      const fraction = this.position - left;
      output[written++] = data[left] + (data[left + 1] - data[left]) * fraction;
      this.position += this.ratio;
    }
    this.position -= data.length - 1;
    this.previous = data[data.length - 1];
    return written === output.length ? output : output.slice(0, written);
  }
}

function buildChannelSelector() {
  CHANNELS.forEach((channel) => {
    const button = document.createElement("button");
    button.className = "channel-button";
    button.type = "button";
    button.role = "radio";
    button.ariaChecked = channel.index === selectedChannel ? "true" : "false";
    button.ariaLabel = `Canal ${channel.index + 1}, ${channel.label}`;
    button.style.setProperty("--angle", `${channel.angle}deg`);
    button.textContent = String(channel.index + 1).padStart(2, "0");
    button.addEventListener("click", () => selectChannel(channel.index));
    elements.buttons.append(button);
  });
}

function selectChannel(index) {
  selectedChannel = index;
  const channel = CHANNELS[index];
  [...elements.buttons.children].forEach((button, buttonIndex) => {
    button.ariaChecked = buttonIndex === index ? "true" : "false";
  });
  elements.angle.value = `${String(channel.angle).padStart(3, "0")}°`;
  elements.number.textContent = `Canal ${index + 1}`;
  elements.position.textContent = channel.label;
  elements.line.style.transform = `rotate(${channel.angle}deg)`;

  if (socket?.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: "select", channel: index }));
  }
  if (workletNode) workletNode.port.postMessage({ type: "reset" });
  if (resampler) resampler.reset();
  drawWaveform(new Float32Array());
}

function setConnection(state, text) {
  elements.connectionDot.className = `status-dot ${state}`;
  elements.connectionText.textContent = text;
}

function connectSocket() {
  clearTimeout(reconnectTimer);
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  socket = new WebSocket(`${protocol}//${location.host}/ws`);
  socket.binaryType = "arraybuffer";
  setConnection("", "Conectando");

  socket.addEventListener("open", () => {
    setConnection("connected", "Web conectada");
    socket.send(JSON.stringify({ type: "select", channel: selectedChannel }));
  });

  socket.addEventListener("message", (event) => {
    if (typeof event.data === "string") {
      const message = JSON.parse(event.data);
      if (message.type === "config") sourceRate = message.sampleRate;
      return;
    }

    const samples = new Float32Array(event.data);
    drawWaveform(samples);
    if (!listening || !workletNode || !resampler) return;
    const converted = resampler.process(samples);
    workletNode.port.postMessage(
      { type: "samples", samples: converted },
      [converted.buffer],
    );
  });

  socket.addEventListener("close", () => {
    setConnection("error", "Reconectando");
    elements.signalState.textContent = "Sin conexión";
    reconnectTimer = setTimeout(connectSocket, 1800);
  });

  socket.addEventListener("error", () => socket.close());
}

async function toggleListening() {
  if (!audioContext) {
    try {
      audioContext = new AudioContext({ latencyHint: "interactive" });
      await audioContext.audioWorklet.addModule("/static/audio-worklet.js");
      workletNode = new AudioWorkletNode(audioContext, "pcm-stream-processor", {
        numberOfInputs: 0,
        numberOfOutputs: 1,
        outputChannelCount: [1],
      });
      workletNode.connect(audioContext.destination);
      resampler = new StreamingResampler(sourceRate, audioContext.sampleRate);
      workletNode.port.onmessage = ({ data }) => {
        if (data.type === "playing") elements.signalState.textContent = "Reproduciendo";
        if (data.type === "underrun") elements.signalState.textContent = "Recargando búfer";
      };
    } catch (error) {
      elements.notice.textContent = `No se pudo iniciar el audio: ${error.message}`;
      return;
    }
  }

  if (!listening) {
    await audioContext.resume();
    listening = true;
    elements.listen.classList.add("active");
    elements.listenLabel.textContent = "Detener escucha";
    elements.notice.textContent = `Salida del navegador: ${audioContext.sampleRate.toLocaleString("es-ES")} Hz`;
  } else {
    listening = false;
    workletNode.port.postMessage({ type: "reset" });
    resampler.reset();
    await audioContext.suspend();
    elements.listen.classList.remove("active");
    elements.listenLabel.textContent = "Activar escucha";
    elements.signalState.textContent = "Escucha detenida";
  }
}

async function updateStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) throw new Error("status");
    const status = await response.json();
    elements.sourceStatus.textContent = status.source_live ? "En directo" : "En espera";
    elements.listenerCount.textContent = status.listeners;
    if (!status.source_live && listening) elements.signalState.textContent = "Esperando señal";
  } catch {
    elements.sourceStatus.textContent = "No disponible";
  }
}

function drawWaveform(samples) {
  const canvas = elements.waveform;
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(canvas.clientWidth * ratio));
  const height = Math.max(1, Math.floor(canvas.clientHeight * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  const context = canvas.getContext("2d");
  context.clearRect(0, 0, width, height);
  context.strokeStyle = "rgba(34, 86, 108, 0.18)";
  context.lineWidth = ratio;
  context.beginPath();
  context.moveTo(0, height / 2);
  context.lineTo(width, height / 2);
  context.stroke();

  if (!samples.length) return;
  context.strokeStyle = "#d75432";
  context.lineWidth = 1.4 * ratio;
  context.beginPath();
  const step = samples.length / width;
  for (let x = 0; x < width; x++) {
    const sample = samples[Math.min(samples.length - 1, Math.floor(x * step))];
    const y = height / 2 - Math.max(-1, Math.min(1, sample)) * height * 0.42;
    if (x === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  }
  context.stroke();
}

buildChannelSelector();
selectChannel(0);
connectSocket();
elements.listen.addEventListener("click", toggleListening);
window.addEventListener("resize", () => drawWaveform(new Float32Array()));
setInterval(updateStatus, 1500);
updateStatus();
