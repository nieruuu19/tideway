import { afterEach, describe, expect, it, vi } from "vitest";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";

(
  globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }
).IS_REACT_ACT_ENVIRONMENT = true;

/**
 * The shell renders the login screen whenever `logged_in` is false, so
 * a failed `/api/auth/status` must not be taken at face value: the dev
 * launcher brings Vite up before uvicorn has bound its port, and the
 * resulting connection error used to paint Login over a perfectly valid
 * session (which then walked the user through a needless re-auth).
 */

vi.mock("@/api/client", () => ({
  api: { auth: { status: vi.fn() } },
}));

const { api } = await import("@/api/client");
const { useAuth } = await import("./useAuth");

const statusMock = api.auth.status as ReturnType<typeof vi.fn>;

let latest: ReturnType<typeof useAuth> | null = null;
function Probe() {
  latest = useAuth();
  return null;
}

let root: Root | null = null;
function mount() {
  const el = document.createElement("div");
  root = createRoot(el);
  act(() => root!.render(<Probe />));
}

afterEach(() => {
  act(() => root?.unmount());
  root = null;
  latest = null;
  vi.useRealTimers();
  statusMock.mockReset();
});

describe("useAuth boot retry", () => {
  it("keeps the spinner instead of reporting signed-out while the backend boots", async () => {
    vi.useFakeTimers();
    statusMock
      .mockRejectedValueOnce(new Error("Failed to fetch"))
      .mockRejectedValueOnce(new Error("502: Bad Gateway"))
      .mockResolvedValue({
        logged_in: true,
        username: "someone",
        avatar: null,
        user_id: "1",
      });

    mount();
    // First attempt failed; the hook is waiting out the boot window.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(latest!.loading).toBe(true);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(250);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(500);
    });

    expect(latest!.loading).toBe(false);
    expect(latest!.logged_in).toBe(true);
    expect(latest!.username).toBe("someone");
  });

  it("settles on signed-out once the retry window is exhausted", async () => {
    vi.useFakeTimers();
    statusMock.mockRejectedValue(new Error("Failed to fetch"));

    mount();
    await act(async () => {
      // 250 + 500 + 1000 + 2000 + 3000 + 4000 + 4000 ms of retries,
      // then give up.
      await vi.advanceTimersByTimeAsync(15000);
    });

    expect(latest!.loading).toBe(false);
    expect(latest!.logged_in).toBe(false);
  });
});
