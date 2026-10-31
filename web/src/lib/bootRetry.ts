/**
 * Retry helper for requests fired the instant the SPA mounts.
 *
 * The dev launcher and uvicorn start together, so the Vite proxy can
 * answer the first requests with ECONNREFUSED while the backend is
 * still importing. Any request that decides durable UI state from its
 * first answer has to ride that window out — for `/api/auth/status` a
 * single failed probe used to mean "signed out", which bounced a
 * signed-in user to the login screen that then retried happily.
 *
 * ~15 s of patience: a cold backend import runs 2-8 s depending on
 * whether its bytecode and dependencies are warm, and the dev launcher
 * adds a --reload supervisor on top. Callers keep showing their boot
 * spinner instead of publishing a "signed out" that was never asked.
 */
const BOOT_RETRY_DELAYS_MS = [250, 500, 1000, 2000, 3000, 4000, 4000];

export async function retryBoot<T>(fn: () => Promise<T>): Promise<T> {
  let lastError: unknown;
  for (let attempt = 0; ; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastError = err;
      const delay = BOOT_RETRY_DELAYS_MS[attempt];
      if (delay === undefined) break;
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }
  throw lastError;
}
