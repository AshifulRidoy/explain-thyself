/**
 * SSE frame parsing for live mode. `EventSource` cannot POST a JSON body,
 * so live traces are fetch-streamed and framed here — chunk boundaries are
 * arbitrary; frames are separated by a blank line.
 *
 * Pure except for the decoder it owns. `push` returns complete frames.
 */

export interface SseFrame {
  event: string;
  data: string;
}

export class SseParser {
  private buffer = "";
  private decoder = new TextDecoder();

  /** Feed a transport chunk; get back every complete frame it completed. */
  push(chunk: Uint8Array | string): SseFrame[] {
    this.buffer += typeof chunk === "string" ? chunk : this.decoder.decode(chunk, { stream: true });
    const frames: SseFrame[] = [];
    // frames end with \n\n (spec also allows \r\n\r\n)
    let boundary = this.buffer.search(/\r?\n\r?\n/);
    while (boundary !== -1) {
      const raw = this.buffer.slice(0, boundary);
      this.buffer = this.buffer.slice(
        boundary + (this.buffer[boundary] === "\r" ? 4 : 2),
      );
      const frame = parseFrame(raw);
      if (frame) frames.push(frame);
      boundary = this.buffer.search(/\r?\n\r?\n/);
    }
    return frames;
  }

  /** Flush any trailing frame that lacked its blank line (server closed). */
  flush(): SseFrame[] {
    const rest = this.buffer;
    this.buffer = "";
    const frame = parseFrame(rest);
    return frame ? [frame] : [];
  }
}

function parseFrame(raw: string): SseFrame | null {
  let event = "message";
  const dataLines: string[] = [];
  for (const line of raw.split(/\r?\n/)) {
    if (line.startsWith(":")) continue; // heartbeat comment
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") event = value;
    else if (field === "data") dataLines.push(value);
  }
  if (dataLines.length === 0) return null;
  return { event, data: dataLines.join("\n") };
}
