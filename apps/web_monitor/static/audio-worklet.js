class PCMStreamProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.offset = 0;
    this.queuedSamples = 0;
    this.primed = false;
    this.targetSamples = Math.round(sampleRate * 0.18);
    this.underruns = 0;

    this.port.onmessage = ({ data }) => {
      if (data.type === "samples") {
        const samples = data.samples;
        this.queue.push(samples);
        this.queuedSamples += samples.length;
        this.trimExcess();
      } else if (data.type === "reset") {
        this.queue = [];
        this.offset = 0;
        this.queuedSamples = 0;
        this.primed = false;
      }
    };
  }

  trimExcess() {
    const maximum = sampleRate * 1.5;
    while (this.queuedSamples > maximum && this.queue.length > 2) {
      const removed = this.queue.shift();
      this.queuedSamples -= removed.length;
      this.offset = 0;
    }
  }

  nextSample() {
    while (this.queue.length) {
      const chunk = this.queue[0];
      if (this.offset < chunk.length) {
        const value = chunk[this.offset++];
        this.queuedSamples--;
        return value;
      }
      this.queue.shift();
      this.offset = 0;
    }
    return null;
  }

  process(_inputs, outputs) {
    const output = outputs[0][0];
    output.fill(0);

    if (!this.primed) {
      if (this.queuedSamples < this.targetSamples) return true;
      this.primed = true;
      this.port.postMessage({ type: "playing" });
    }

    for (let index = 0; index < output.length; index++) {
      const value = this.nextSample();
      if (value === null) {
        this.primed = false;
        this.underruns++;
        this.port.postMessage({ type: "underrun", count: this.underruns });
        break;
      }
      output[index] = value;
    }
    return true;
  }
}

registerProcessor("pcm-stream-processor", PCMStreamProcessor);
