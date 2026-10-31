/**
 * Tests for the playback-controller lease. The lease is what keeps two
 * open Tideway pages from both advancing on `ended` with their own
 * queues — the "playlist jumps to unrelated songs / wrong track flashes
 * at every boundary" bug.
 */
import { describe, expect, it } from "vitest";
import {
  LEASE_TTL_MS,
  PLAYER_LEASE_KEY,
  PlayerLease,
  type LeaseStore,
} from "./usePlayerOwnership";

function fakeStore(): LeaseStore & { items: Map<string, string> } {
  const items = new Map<string, string>();
  return {
    items,
    getItem: (key) => items.get(key) ?? null,
    setItem: (key, value) => void items.set(key, value),
    removeItem: (key) => void items.delete(key),
  };
}

function clock(start = 1_000_000) {
  let now = start;
  return {
    now: () => now,
    advance: (ms: number) => {
      now += ms;
    },
  };
}

describe("PlayerLease", () => {
  it("takes a free lease on the first heartbeat", () => {
    const store = fakeStore();
    const lease = new PlayerLease(store, "page-a", clock().now);

    expect(lease.isOwner).toBe(false);
    expect(lease.refresh()).toBe(true);
    expect(lease.isOwner).toBe(true);
    expect(JSON.parse(store.items.get(PLAYER_LEASE_KEY)!)).toMatchObject({
      id: "page-a",
    });
  });

  it("yields to a live holder and does not overwrite its lease", () => {
    const store = fakeStore();
    const time = clock();
    const holder = new PlayerLease(store, "page-a", time.now);
    holder.refresh();
    const original = store.items.get(PLAYER_LEASE_KEY);

    const viewer = new PlayerLease(store, "page-b", time.now);
    expect(viewer.refresh()).toBe(false);
    expect(viewer.isOwner).toBe(false);
    expect(store.items.get(PLAYER_LEASE_KEY)).toBe(original);
  });

  it("takes over once the holder's lease expires", () => {
    const store = fakeStore();
    const time = clock();
    const holder = new PlayerLease(store, "page-a", time.now);
    holder.refresh();

    const viewer = new PlayerLease(store, "page-b", time.now);
    time.advance(LEASE_TTL_MS + 1);

    expect(viewer.refresh()).toBe(true);
    expect(viewer.isOwner).toBe(true);
    expect(JSON.parse(store.items.get(PLAYER_LEASE_KEY)!).id).toBe("page-b");
  });

  it("lets the holder extend its own lease past the ttl", () => {
    const store = fakeStore();
    const time = clock();
    const holder = new PlayerLease(store, "page-a", time.now);
    holder.refresh();

    time.advance(LEASE_TTL_MS - 1);
    expect(holder.refresh()).toBe(true);

    const viewer = new PlayerLease(store, "page-b", time.now);
    expect(viewer.refresh()).toBe(false);
    expect(holder.isOwner).toBe(true);
  });

  it("claim() steals an active lease for an explicit user action", () => {
    const store = fakeStore();
    const time = clock();
    const holder = new PlayerLease(store, "page-a", time.now);
    holder.refresh();

    const viewer = new PlayerLease(store, "page-b", time.now);
    viewer.claim();

    expect(viewer.isOwner).toBe(true);
    expect(JSON.parse(store.items.get(PLAYER_LEASE_KEY)!).id).toBe("page-b");

    holder.refresh();
    expect(holder.isOwner).toBe(false);
  });

  it("release() leaves another page's lease alone", () => {
    const store = fakeStore();
    const time = clock();
    const holder = new PlayerLease(store, "page-a", time.now);
    holder.refresh();
    const original = store.items.get(PLAYER_LEASE_KEY);

    const bystander = new PlayerLease(store, "page-b", time.now);
    bystander.release();
    expect(store.items.get(PLAYER_LEASE_KEY)).toBe(original);

    holder.release();
    expect(store.items.has(PLAYER_LEASE_KEY)).toBe(false);
    expect(holder.isOwner).toBe(false);
  });

  it("treats a malformed lease value as free", () => {
    const store = fakeStore();
    store.items.set(PLAYER_LEASE_KEY, "{not json");
    const lease = new PlayerLease(store, "page-a", clock().now);

    expect(lease.refresh()).toBe(true);
    expect(lease.isOwner).toBe(true);
  });
});
