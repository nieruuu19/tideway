import { useCallback, useEffect, useRef, type MutableRefObject } from "react";

/**
 * Playback-controller lease.
 *
 * Tideway can be open in more than one page at once — a duplicated tab,
 * or the app window plus a browser tab on 127.0.0.1:8000. Every page
 * holds its own queue and reacts to the same SSE stream, so without
 * arbitration each page advances on `ended` with its own "next" track,
 * preloads its own successor, and overwrites the persisted now-playing
 * snapshot. Observed symptom: at every track boundary the wrong track
 * plays for a fraction of a second before the right one, and a playlist
 * session eventually drifts into the other page's radio queue.
 *
 * One page holds this lease at a time. Only the holder advances the
 * queue, fires preloads and persists state; the others stay viewers and
 * mirror the holder's queue (see the storage listener in `usePlayer`).
 * A page takes the lease when it mounts and the lease is free/expired,
 * and unconditionally when the user touches its transport controls.
 */
export const PLAYER_LEASE_KEY = "tideway:player-controller";
export const LEASE_TTL_MS = 5000;
const HEARTBEAT_MS = 1500;

interface Lease {
  id: string;
  exp: number;
}

export interface LeaseStore {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

const memoryStore: LeaseStore = (() => {
  const items = new Map<string, string>();
  return {
    getItem: (key) => items.get(key) ?? null,
    setItem: (key, value) => void items.set(key, value),
    removeItem: (key) => void items.delete(key),
  };
})();

function resolveStore(): LeaseStore {
  try {
    // Private-mode / disabled storage throws on access, not just write.
    return window.localStorage;
  } catch {
    return memoryStore;
  }
}

function makeId(): string {
  const c = globalThis.crypto as Crypto | undefined;
  if (c && typeof c.randomUUID === "function") return c.randomUUID();
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

/** Pure lease logic, injectable clock/store so it can be unit tested. */
export class PlayerLease {
  private owned = false;

  constructor(
    private readonly store: LeaseStore,
    readonly id: string,
    private readonly now: () => number = Date.now,
  ) {}

  get isOwner(): boolean {
    return this.owned;
  }

  private read(): Lease | null {
    try {
      const raw = this.store.getItem(PLAYER_LEASE_KEY);
      if (!raw) return null;
      const parsed = JSON.parse(raw) as Partial<Lease>;
      if (typeof parsed.id !== "string" || typeof parsed.exp !== "number") {
        return null;
      }
      return { id: parsed.id, exp: parsed.exp };
    } catch {
      return null;
    }
  }

  private write(): void {
    try {
      this.store.setItem(
        PLAYER_LEASE_KEY,
        JSON.stringify({ id: this.id, exp: this.now() + LEASE_TTL_MS }),
      );
    } catch {
      /* storage full or disabled — keep the in-memory answer */
    }
  }

  /** Refresh while held; take a free or expired lease; yield otherwise. */
  refresh(): boolean {
    const cur = this.read();
    if (cur && cur.id !== this.id && cur.exp > this.now()) {
      this.owned = false;
      return false;
    }
    this.write();
    this.owned = true;
    return true;
  }

  /** Take the lease even if another page holds it (explicit user action). */
  claim(): void {
    this.write();
    this.owned = true;
  }

  /** Drop the lease so another page can take over immediately. */
  release(): void {
    const cur = this.read();
    if (cur && cur.id === this.id) {
      try {
        this.store.removeItem(PLAYER_LEASE_KEY);
      } catch {
        /* ignore */
      }
    }
    this.owned = false;
  }
}

export function usePlayerOwnership(): {
  isOwnerRef: MutableRefObject<boolean>;
  claimOwnership: () => void;
} {
  const isOwnerRef = useRef(false);
  const leaseRef = useRef<PlayerLease | null>(null);
  if (leaseRef.current === null) {
    leaseRef.current = new PlayerLease(resolveStore(), makeId());
  }

  const sync = useCallback(() => {
    const lease = leaseRef.current;
    if (!lease) return;
    lease.refresh();
    isOwnerRef.current = lease.isOwner;
  }, []);

  const claimOwnership = useCallback(() => {
    const lease = leaseRef.current;
    if (!lease) return;
    lease.claim();
    isOwnerRef.current = true;
  }, []);

  useEffect(() => {
    sync();
    const timer = window.setInterval(sync, HEARTBEAT_MS);
    const onStorage = (e: StorageEvent) => {
      if (e.key === PLAYER_LEASE_KEY) sync();
    };
    // Release on the way out so the other page takes over without
    // waiting out the TTL. `isOwnerRef` stays true: the persistence
    // flush that also runs on pagehide still needs to write.
    const onHide = () => leaseRef.current?.release();
    const onShow = () => sync();
    window.addEventListener("storage", onStorage);
    window.addEventListener("pagehide", onHide);
    window.addEventListener("pageshow", onShow);
    return () => {
      window.clearInterval(timer);
      window.removeEventListener("storage", onStorage);
      window.removeEventListener("pagehide", onHide);
      window.removeEventListener("pageshow", onShow);
      leaseRef.current?.release();
      isOwnerRef.current = false;
    };
  }, [sync]);

  return { isOwnerRef, claimOwnership };
}
