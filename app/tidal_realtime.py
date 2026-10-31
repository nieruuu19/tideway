"""Tidal realtime listener: pause Tideway when another device on the
same Tidal account starts playing.

Tidal's official clients react to playback events from other devices
via a realtime bus. When another device on the same account begins a
playback session, the bus pushes a state-change frame and the
receiving client pauses locally. Spotify and the official Tidal app
both behave this way out of the box; users coming from those expect
Tideway to behave the same.

The send half of this contract already exists. `app/play_reporter.py`
posts our own `playback_session` events to the event-producer bus
(`ec.tidal.com/api/event-batch`), which is what makes "Tideway plays
show up in Recently Played" and "starting Tideway pauses other
devices that listen to the same bus" work today. This module is the
**receive** half: a long-lived WebSocket subscription to Tidal's
realtime bus ("Pushkin"), with an `on_other_device_started` callback
that the launcher wires to `PCMPlayer.pause()`.

## Protocol

Reverse-engineered from the Tidal web client's `pushkin-*.js` module
(2026-05-20). Two phases:

  1. **Token acquisition.** POST to `https://api.tidal.com/v1/rt/
     connect` with `Authorization: Bearer <access_token>`. Response
     is `{"url": "wss://pushkin-v2.tidal.com/public/token/<uuid>/
     ws"}` — the full WebSocket URL including a per-session token
     baked into the path. Tokens are minted per connection; we
     re-acquire on every (re)connect.

  2. **WebSocket session.** Plain JSON frames. Incoming types we
     handle:
       - `PRIVILEGED_SESSION_NOTIFICATION` — payload has
         `clientDisplayName` (e.g. "iOS") and `sessionId`. Fires
         when another device on the same Tidal account takes
         playback. We pause local playback in response.
       - `RECONNECT` — server-initiated reconnect. Close and
         reopen.
       - Anything else: log and ignore.

     Outgoing frames are optional. The web client only sends
     `USER_ACTION` messages in response to page user-action events
     (clicks, keypresses); they're not heartbeats and Tidal's
     backend doesn't seem to require them for connection liveness,
     so we don't send anything.

## Threading model

The listener runs inside FastAPI's asyncio event loop as a background
task spawned from lifespan startup. The task lifecycle is:

  start() → asyncio.create_task → _run() loop:
      while not shutdown:
          try: await _connect_and_serve()
          except: log + backoff
          await sleep(backoff)

Cancellation on shutdown is cooperative: lifespan teardown calls
stop(), which sets a shutdown flag and cancels the task; _run() sees
the cancellation, closes the WebSocket if open, and exits.

The pause callback fires from inside the asyncio loop. PCMPlayer's
methods are thread-safe (they take `_lock` internally) so calling
into them from the loop is fine; we don't need a thread hop.

## Self-event filtering

The web client's frame handler pauses unconditionally on every
`PRIVILEGED_SESSION_NOTIFICATION`. That works for the web client
because it is an observer, but Tideway both sends its own playback
events (`play_reporter`) and sends `USER_ACTION` when local playback
starts. Tidal's backend echoes that back to us as a
`PRIVILEGED_SESSION_NOTIFICATION` whose `clientDisplayName` is our own
client (e.g. "Android Automotive"), which used to make Tideway pause
itself the moment it started playing.

Two identifiers are used, because Tidal does not always echo the same
one back:

  - The access token's JWT carries our OAuth session id in the `sid`
    claim, which is what an echo of our Pushkin session uses. The
    claim rotates when the token refreshes, so callers should hand us
    every sid observed during the process lifetime, not just the
    current one.
  - The `playbackSessionId` UUID our `play_reporter` stamps on its
    events. Tidal's event pipeline learns that id from our own play
    reports and can echo it back as a session notification, in which
    case `sessionId` has nothing to do with our JWT.

A notification whose `sessionId` (or `playbackSessionId`) is one of
ours is counted in `self_events_ignored` and never reaches the pause
callback. Both providers are optional: without them every notification
is treated as another device's, which is the old behavior.

The most recent notifications are kept in `ListenerStatus` (display
name, ids, whether they matched) so a self-echo that slips through the
filter can be diagnosed from `/api/realtime/status` instead of
guessing.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Iterable, Optional

log = logging.getLogger(__name__)


# HTTP endpoint that mints a per-session WebSocket URL. Returns
# {"url": "wss://pushkin-v2.tidal.com/public/token/<uuid>/ws"}.
# Reverse-engineered from `pushkin-*.js`'s `l()` function.
_RT_CONNECT_URL = "https://api.tidal.com/v1/rt/connect"

# Backoff schedule for reconnects. Doubles each consecutive failure,
# capped so we don't drift into multi-minute holes that delay the
# user's first cross-device-pause event after a network blip. The
# first failure waits half a second; the cap of 60s keeps us
# responsive while still being polite to Tidal's bus during outages.
_BACKOFF_INITIAL_SEC = 0.5
_BACKOFF_MAX_SEC = 60.0

# How long to wait for the token POST + WebSocket handshake before
# giving up and treating it as a connection failure. The token POST
# is a single HTTPS round-trip against api.tidal.com; the handshake
# is a single TLS+WS upgrade. Tens of seconds is generous; longer
# than that means something's wrong (DNS, blocked, captive portal)
# and we should back off, not wait.
_CONNECT_TIMEOUT_SEC = 10.0

# How many PRIVILEGED_SESSION_NOTIFICATION frames to keep for the
# `/api/realtime/status` diagnostic. Small on purpose: this is a
# debugging aid, not an event log.
_RECENT_NOTIFICATIONS_CAP = 8


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

# Function the listener calls to obtain the current Tidal access token.
# Returning None means "no session right now" and the listener will
# back off. Tokens rotate on refresh, so the listener calls this every
# time it (re)connects rather than caching once.
TokenProvider = Callable[[], Optional[str]]

# Function the listener calls to obtain the current Tidal session id
# (the `sid` claim of the access token). Used to recognise Pushkin
# echoing our own playback events back at us. Returning None means
# "can't tell", in which case notifications are delivered as before.
SessionIdProvider = Callable[[], Optional[str]]

# Function returning every session id that could belong to us: the
# current token's `sid`, any `sid` seen earlier in the process (it
# rotates on refresh), and the `playbackSessionId` UUIDs of our recent
# play reports. Preferred over `session_id_provider`; both are honoured
# when both are supplied.
OwnSessionIdsProvider = Callable[[], Iterable[str]]

# Callback fired when another device on the same account starts
# playing. The desktop launcher binds this to PCMPlayer.pause(). The
# callback runs on the asyncio loop and may be sync or async.
OtherDeviceStartedCallback = Callable[[dict], Awaitable[None] | None]


@dataclass
class ListenerStatus:
    """What `/api/realtime/status` returns. Diagnostic surface only."""

    # One of: "idle", "connecting", "connected", "reconnecting",
    # "stopped". "idle" means the listener was constructed but
    # start() hasn't been called.
    phase: str = "idle"
    # Last error message, if any. Cleared on successful connect.
    last_error: Optional[str] = None
    # How many times the listener has reconnected since start(). 0
    # is a fresh listener that hasn't reconnected yet.
    reconnect_count: int = 0
    # How many "other device started" events we've delivered to the
    # callback since start(). Useful sanity check that the wire-up
    # is working when capturing fresh fixtures.
    events_received: int = 0
    # How many notifications were dropped because their sessionId
    # matched our own token's session id — i.e. Tidal echoing our own
    # playback back at us. A non-zero value next to events_received=0
    # means the self-filter is doing its job.
    self_events_ignored: int = 0
    # The most recent PRIVILEGED_SESSION_NOTIFICATION frames, oldest
    # first. Each entry has `client_display_name`, `session_id`,
    # `playback_session_id`, `payload_keys` and `matched_own_id`.
    # Diagnostic only — lets `/api/realtime/status` show exactly which
    # id Tidal echoed when a false cross-device pause is reported.
    recent_notifications: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Listener
# ---------------------------------------------------------------------------


@dataclass
class RealtimeListener:
    """Long-lived connection to Tidal's realtime bus.

    Construct with a token provider and an event callback. Call
    start() to spawn the connection task; call stop() to cancel it.
    Status is observable via status() at any time.
    """

    token_provider: TokenProvider
    on_other_device_started: OtherDeviceStartedCallback
    # Optional. Without it, every PRIVILEGED_SESSION_NOTIFICATION is
    # treated as another device's (no self-filter).
    session_id_provider: Optional[SessionIdProvider] = None
    # Optional. Returns all ids that could identify our own session.
    # Preferred over `session_id_provider`; when both are supplied the
    # ids are merged.
    own_session_ids_provider: Optional[OwnSessionIdsProvider] = None

    # Internal state. Not user-tunable.
    _task: Optional[asyncio.Task] = field(default=None, init=False)
    _shutdown: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    _status: ListenerStatus = field(default_factory=ListenerStatus, init=False)
    # Stashed when start() runs so signal_user_action_sync() can
    # hand the coroutine back to the right loop from sync code (the
    # PCMPlayer state-change listener fires from arbitrary threads).
    _loop: Optional[asyncio.AbstractEventLoop] = field(default=None, init=False)
    # The currently-open WebSocket while _connect_and_serve is in
    # the middle of its serve loop, else None. Read by
    # send_user_action; never written from outside the loop.
    _ws: Optional[object] = field(default=None, init=False)
    # UUIDs baked into the Pushkin connection URLs we've minted, newest
    # last. Tidal's bus identifies the originating connection as the
    # privileged session; an echo of our own USER_ACTION can carry this
    # id instead of the OAuth `sid`.
    _own_pushkin_ids: list[str] = field(default_factory=list, init=False)

    @property
    def is_protocol_known(self) -> bool:
        """Reverse-engineered protocol is in place since 2026-05-20;
        kept on the public surface for the `/api/realtime/status`
        diagnostic endpoint and for any callers that want to gate
        on it. Always True now."""
        return True

    def start(self) -> None:
        """Spawn the connection task. Idempotent: a second call while
        already running is a no-op."""
        if self._task is not None and not self._task.done():
            return
        self._shutdown.clear()
        self._status = ListenerStatus(phase="connecting")
        loop = asyncio.get_running_loop()
        self._loop = loop
        self._task = loop.create_task(
            self._run(), name="tidal-realtime-listener"
        )

    def stop(self) -> None:
        """Cancel the connection task. Safe to call multiple times."""
        self._shutdown.set()
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._status.phase = "stopped"

    def status(self) -> ListenerStatus:
        """Snapshot of the listener's current state. The dataclass is
        frozen at the moment of read; callers don't need to lock."""
        return ListenerStatus(
            phase=self._status.phase,
            last_error=self._status.last_error,
            reconnect_count=self._status.reconnect_count,
            events_received=self._status.events_received,
            self_events_ignored=self._status.self_events_ignored,
            recent_notifications=list(self._status.recent_notifications),
        )

    # -- internals ----------------------------------------------------------

    async def _run(self) -> None:
        """Main connect-and-serve loop. Reconnects on drop with
        exponential backoff. Exits when _shutdown is set."""
        backoff = _BACKOFF_INITIAL_SEC
        first_pass = True
        while not self._shutdown.is_set():
            if not first_pass:
                self._status.phase = "reconnecting"
                self._status.reconnect_count += 1
            first_pass = False
            try:
                await self._connect_and_serve()
                # Clean exit means the server closed without an error;
                # treat as a brief blip and reconnect promptly.
                backoff = _BACKOFF_INITIAL_SEC
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self._status.last_error = repr(exc)
                log.warning(
                    "tidal_realtime: connection error; backing off %.1fs: %r",
                    backoff, exc,
                )
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(), timeout=backoff
                )
                # Shutdown signalled mid-backoff; exit cleanly.
                return
            except asyncio.TimeoutError:
                pass
            backoff = min(backoff * 2.0, _BACKOFF_MAX_SEC)

    async def _connect_and_serve(self) -> None:
        """Acquire a token, open the Pushkin WebSocket, and serve
        frames until the connection drops. Raised exceptions trigger
        reconnect via _run().

        Lazy import of aiohttp + websockets so test environments that
        construct the listener without an event loop don't fail at
        import time on a missing optional dep.
        """
        access_token = self.token_provider()
        if not access_token:
            # No Tidal session right now (logged out, refresh in
            # flight). Raise so _run() backs off and tries again
            # once the session is back.
            raise RuntimeError("no Tidal access token available")

        import ssl

        import aiohttp
        import certifi
        import websockets

        # Use certifi's CA bundle explicitly. The python.org Python
        # macOS distribution doesn't trust the system Keychain by
        # default, so aiohttp's default SSL context fails to verify
        # the api.tidal.com cert with "unable to get local issuer
        # certificate". The existing `requests`-based code in
        # _fetch_latest_release dodges this because requests ships
        # certifi; aiohttp has to be pointed at it.
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())

        # Phase 1: POST to /v1/rt/connect with the Bearer token to
        # mint a per-session WebSocket URL. The response carries the
        # full wss:// URL including the token UUID baked into the
        # path; we don't need to construct it ourselves.
        connector = aiohttp.TCPConnector(ssl=ssl_ctx)
        async with aiohttp.ClientSession(connector=connector) as http:
            async with http.post(
                _RT_CONNECT_URL,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=_CONNECT_TIMEOUT_SEC),
            ) as resp:
                if resp.status == 401:
                    # Token rejected — the access_token from the
                    # provider is stale. Force a refresh next time
                    # by surfacing this as a specific error.
                    raise RuntimeError(
                        "Pushkin token request returned 401; access "
                        "token rejected"
                    )
                resp.raise_for_status()
                payload = await resp.json()
        ws_url = payload.get("url")
        if not ws_url:
            raise RuntimeError(
                f"Pushkin token response missing 'url': {payload!r}"
            )
        self._remember_pushkin_id(ws_url)

        # Phase 2: open the WebSocket and serve frames. No subscribe
        # message and no periodic client heartbeats. The outbound
        # USER_ACTION frame is sent on demand by
        # signal_user_action_sync() — it's how we tell Pushkin
        # "this device is now the active one" so other devices on
        # the same account pause in response. Same certifi CA
        # bundle as the token POST so the WS handshake doesn't trip
        # the same system-CA issue.
        async with websockets.connect(
            ws_url,
            open_timeout=_CONNECT_TIMEOUT_SEC,
            ssl=ssl_ctx,
        ) as ws:
            self._ws = ws
            self._status.phase = "connected"
            self._status.last_error = None
            log.info(
                "tidal_realtime: connected to Pushkin (reconnect_count=%d)",
                self._status.reconnect_count,
            )
            try:
                async for raw in ws:
                    frame = self._parse_frame(raw)
                    if frame is None:
                        continue
                    await self._dispatch_frame(frame, ws)
            finally:
                # Clear the ws reference so signal_user_action_sync
                # callers in flight stop trying to send through a
                # half-closed socket. The reconnect loop will
                # re-populate this on the next successful connect.
                self._ws = None

    def _remember_pushkin_id(self, ws_url: str) -> None:
        """Track the connection id embedded in the minted WS URL."""
        match = re.search(r"/token/([^/?#]+)", ws_url)
        if not match:
            return
        connection_id = match.group(1).strip().lower()
        if not connection_id:
            return
        ids = self._own_pushkin_ids
        if connection_id not in ids:
            ids.append(connection_id)
        # A handful is plenty: notifications for a connection that was
        # replaced are stale anyway.
        del ids[:-8]

    def _own_session_id(self) -> Optional[str]:
        """Our current Tidal session id, or None when unknown.

        Best-effort: a provider that raises or returns garbage must
        not break the receive loop. Treating it as unknown falls back
        to delivering the notification.
        """
        provider = self.session_id_provider
        if provider is None:
            return None
        try:
            value = provider()
        except Exception:
            log.exception("tidal_realtime: session_id_provider raised")
            return None
        return str(value) if value else None

    def _own_session_ids(self) -> set[str]:
        """Every id that could be ours, normalised for comparison.

        Merges the legacy single-id provider with the plural one so
        callers can migrate without breaking older wiring. Failures are
        treated as "no ids": a broken provider must not turn into a
        pause storm.
        """
        ids: set[str] = set()
        current = self._own_session_id()
        if current:
            ids.add(current.strip().lower())
        ids.update(self._own_pushkin_ids)
        provider = self.own_session_ids_provider
        if provider is not None:
            try:
                values = provider() or ()
            except Exception:
                log.exception("tidal_realtime: own_session_ids_provider raised")
                values = ()
            for value in values:
                text = str(value).strip().lower() if value else ""
                if text:
                    ids.add(text)
        return ids

    def _record_notification(
        self,
        payload: dict,
        *,
        matched_own_id: bool,
        own_id_count: int,
    ) -> None:
        """Remember the last few notifications for `/api/realtime/status`.

        Values here are opaque ids and Tidal's own display name — no
        user content — and the endpoint is loopback-only, so keeping
        them is safe and is what makes a future false pause
        diagnosable without reproducing it.
        """
        entry = {
            "ts": int(time.time() * 1000),
            "client_display_name": payload.get("clientDisplayName"),
            "session_id": payload.get("sessionId"),
            "playback_session_id": payload.get("playbackSessionId"),
            "payload_keys": sorted(payload),
            "matched_own_id": matched_own_id,
            "own_id_count": own_id_count,
        }
        recent = self._status.recent_notifications
        recent.append(entry)
        del recent[:-_RECENT_NOTIFICATIONS_CAP]

    async def _dispatch_frame(self, frame: dict, ws) -> None:
        """Route a parsed frame to the right action.

        - PRIVILEGED_SESSION_NOTIFICATION → callback (pause local).
        - RECONNECT → close socket; _run()'s outer loop reopens.
        - anything else → log + ignore (forward-compat).
        """
        frame_type = frame.get("type")
        if frame_type == "PRIVILEGED_SESSION_NOTIFICATION":
            payload = frame.get("payload") or {}
            own_ids = self._own_session_ids()
            matched = None
            for key in ("sessionId", "playbackSessionId"):
                value = payload.get(key)
                if value and str(value).strip().lower() in own_ids:
                    matched = str(value)
                    break
            self._record_notification(
                payload,
                matched_own_id=matched is not None,
                own_id_count=len(own_ids),
            )
            if matched:
                self._status.self_events_ignored += 1
                log.info(
                    "tidal_realtime: ignoring our own session "
                    "notification (id=%s, ours=%d tracked); not pausing",
                    matched,
                    len(own_ids),
                )
                return
            self._status.events_received += 1
            log.info(
                "tidal_realtime: PRIVILEGED_SESSION_NOTIFICATION "
                "from %s (session_id=%s, playback_session_id=%s); "
                "firing pause callback",
                payload.get("clientDisplayName") or "<unknown>",
                payload.get("sessionId"),
                payload.get("playbackSessionId"),
            )
            try:
                result = self.on_other_device_started(payload)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                log.exception(
                    "tidal_realtime: on_other_device_started "
                    "callback raised"
                )
        elif frame_type == "RECONNECT":
            log.info("tidal_realtime: server-initiated reconnect")
            await ws.close()
        else:
            log.warning(
                "tidal_realtime: unhandled frame type %r", frame_type
            )

    async def send_user_action(self) -> bool:
        """Send a USER_ACTION frame on the current WebSocket.

        The Tidal web client fires this whenever the user clicks
        or presses a key in the page; Pushkin's backend uses these
        to mark the device as the "active" one. When Pushkin sees
        a fresh user-action from device B while device A is the
        current active device, it pushes
        PRIVILEGED_SESSION_NOTIFICATION to device A so A pauses.

        For Tideway, the natural trigger is "track started
        playing" — that's the moral equivalent of clicking the
        web app's play button. server.py subscribes to PCMPlayer
        state changes and calls signal_user_action_sync() on the
        transition into the playing state.

        Returns True if the frame was sent, False if there was no
        open connection (caller's fault for racing the reconnect
        loop). Send failures (broken pipe, etc.) raise; the
        receive loop will see them too and trigger reconnect.
        """
        ws = self._ws
        if ws is None:
            log.debug(
                "tidal_realtime: signal_user_action with no open "
                "WebSocket; dropping"
            )
            return False
        # `startedAt` is a Unix-millisecond timestamp; the web
        # client uses a TrueTime-corrected clock (sync'd against
        # api.tidal.com/v1/ping's Date header) but ordinary local
        # time is close enough for Pushkin's purposes — the
        # backend doesn't seem to use this value to break ties.
        body = json.dumps(
            {
                "payload": {"startedAt": int(time.time() * 1000)},
                "type": "USER_ACTION",
            },
            separators=(",", ":"),
        )
        await ws.send(body)
        return True

    def signal_user_action_sync(self) -> None:
        """Sync-friendly wrapper around send_user_action() so the
        PCMPlayer's state-change listener (which runs on player
        threads, not the asyncio loop) can fire USER_ACTION
        without an awaiting context.

        Best-effort: silently drops if the listener isn't running
        yet, the loop isn't available, or the send fails. None of
        those are fatal — the next state change will retry, and
        the worst-case user impact is that one play action doesn't
        pause other devices.
        """
        loop = self._loop
        if loop is None or self._ws is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self.send_user_action(), loop
            )
        except RuntimeError as exc:
            # Loop closed mid-call (shutdown race). Nothing we can
            # do; drop the signal.
            log.debug("tidal_realtime: signal dropped: %r", exc)

    @staticmethod
    def _parse_frame(raw: bytes | str) -> Optional[dict]:
        """Decode a single bus frame. Returns the parsed dict, or
        None if the frame isn't a JSON object we can route on.
        Garbage frames are non-fatal: log and skip."""
        try:
            text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
            obj = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            log.warning("tidal_realtime: dropped non-JSON frame: %r", exc)
            return None
        if not isinstance(obj, dict):
            log.warning(
                "tidal_realtime: dropped non-object frame: %r", obj
            )
            return None
        return obj


# ---------------------------------------------------------------------------
# Module-level singleton + start hook
# ---------------------------------------------------------------------------

_listener: Optional[RealtimeListener] = None


def get_listener() -> Optional[RealtimeListener]:
    """Return the module-level listener if one was started, else None."""
    return _listener


def start_listener(
    token_provider: TokenProvider,
    on_other_device_started: OtherDeviceStartedCallback,
    session_id_provider: Optional[SessionIdProvider] = None,
    own_session_ids_provider: Optional[OwnSessionIdsProvider] = None,
) -> RealtimeListener:
    """Construct (or reuse) the module-level listener and start it.

    Idempotent. Subsequent calls return the existing listener; the
    token_provider and callback from the first call win. Lifespan
    startup is the only intended caller.
    """
    global _listener
    if _listener is None:
        _listener = RealtimeListener(
            token_provider=token_provider,
            on_other_device_started=on_other_device_started,
            session_id_provider=session_id_provider,
            own_session_ids_provider=own_session_ids_provider,
        )
    _listener.start()
    return _listener


def stop_listener() -> None:
    """Shut down the module-level listener if it was started.
    Idempotent."""
    if _listener is not None:
        _listener.stop()
