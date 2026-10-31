import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";

(
  globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }
).IS_REACT_ACT_ENVIRONMENT = true;

import { SeekBar } from "./SeekBar";

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

async function mount(props: {
  currentTime: number;
  duration: number;
  onSeek: (seconds: number) => void;
}) {
  await act(async () => {
    root.render(<SeekBar {...props} />);
  });
  return container.querySelector("input") as HTMLInputElement;
}

/** Write the range value the way a real drag/keyboard does: through the
 *  native setter, then a bubbling `input` event, so React's value tracker
 *  sees a change and fires onChange. */
function setValue(input: HTMLInputElement, value: number) {
  const setter = Object.getOwnPropertyDescriptor(
    HTMLInputElement.prototype,
    "value",
  )!.set!;
  act(() => {
    setter.call(input, String(value));
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
}

function pointer(input: HTMLInputElement, type: "pointerdown" | "pointerup") {
  act(() => {
    input.dispatchEvent(
      new Event(type, { bubbles: true }) as unknown as PointerEvent,
    );
  });
}

function labels() {
  return Array.from(container.querySelectorAll("span")).map(
    (s) => s.textContent,
  );
}

describe("SeekBar drag", () => {
  it("sends one seek for the whole drag, on release", async () => {
    const onSeek = vi.fn();
    const input = await mount({ currentTime: 10, duration: 272, onSeek });

    pointer(input, "pointerdown");
    setValue(input, 40);
    setValue(input, 90);
    setValue(input, 180);
    // Still dragging: the backend must not have been asked to seek yet.
    expect(onSeek).not.toHaveBeenCalled();

    pointer(input, "pointerup");
    expect(onSeek).toHaveBeenCalledTimes(1);
    expect(onSeek).toHaveBeenCalledWith(180);
  });

  it("shows the dragged position while scrubbing, then the real one again", async () => {
    const onSeek = vi.fn();
    const input = await mount({ currentTime: 10, duration: 272, onSeek });

    pointer(input, "pointerdown");
    setValue(input, 150);
    expect(labels()[0]).toBe("2:30");

    pointer(input, "pointerup");
    expect(labels()[0]).toBe("0:10");
  });

  it("a cancelled gesture drops the preview instead of seeking", async () => {
    const onSeek = vi.fn();
    const input = await mount({ currentTime: 10, duration: 272, onSeek });

    pointer(input, "pointerdown");
    setValue(input, 200);
    act(() => {
      input.dispatchEvent(new Event("pointercancel", { bubbles: true }));
    });

    expect(onSeek).not.toHaveBeenCalled();
    expect(labels()[0]).toBe("0:10");
  });

  it("the trailing native change does not undo the committed seek", async () => {
    // Chromium fires one `change` after the gesture, carrying the value
    // the input was re-rendered to. Treating it as another seek sent
    // playback back to where it started.
    const onSeek = vi.fn();
    const input = await mount({ currentTime: 10, duration: 272, onSeek });

    pointer(input, "pointerdown");
    setValue(input, 180);
    pointer(input, "pointerup");
    expect(onSeek).toHaveBeenCalledTimes(1);

    act(() => {
      input.dispatchEvent(new Event("change", { bubbles: true }));
    });
    expect(onSeek).toHaveBeenCalledTimes(1);
    expect(onSeek).toHaveBeenCalledWith(180);
  });

  it("follows the live position after a seek lands", async () => {
    const onSeek = vi.fn();
    await mount({ currentTime: 10, duration: 272, onSeek });
    const input = container.querySelector("input") as HTMLInputElement;

    await act(async () => {
      root.render(<SeekBar currentTime={200} duration={272} onSeek={onSeek} />);
    });

    expect(input.value).toBe("200");
  });
});

describe("SeekBar keyboard", () => {
  it("seeks immediately for a change with no pointer gesture", async () => {
    const onSeek = vi.fn();
    const input = await mount({ currentTime: 10, duration: 272, onSeek });

    // Arrow key on a focused range input: one `input` event, no pointer
    // down to wait for.
    setValue(input, 11);
    expect(onSeek).toHaveBeenCalledTimes(1);
    expect(onSeek).toHaveBeenCalledWith(11);

    setValue(input, 12);
    expect(onSeek).toHaveBeenCalledTimes(2);
    expect(onSeek).toHaveBeenLastCalledWith(12);
  });

  it("keeps seeking per press after a drag has ended", async () => {
    const onSeek = vi.fn();
    const input = await mount({ currentTime: 10, duration: 272, onSeek });

    pointer(input, "pointerdown");
    setValue(input, 100);
    pointer(input, "pointerup");
    expect(onSeek).toHaveBeenCalledTimes(1);

    setValue(input, 101);
    expect(onSeek).toHaveBeenCalledTimes(2);
    expect(onSeek).toHaveBeenLastCalledWith(101);
  });
});
