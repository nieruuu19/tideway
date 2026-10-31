import { useCallback, useEffect, useState } from "react";
import { api } from "@/api/client";
import { clearApiCache } from "@/hooks/useApi";
import { retryBoot } from "@/lib/bootRetry";
import type { AuthStatus } from "@/api/types";

export function useAuth() {
  const [state, setState] = useState<AuthStatus & { loading: boolean }>({
    logged_in: false,
    username: null,
    avatar: null,
    user_id: null,
    loading: true,
  });

  const refresh = useCallback(async () => {
    try {
      // Ride out the launch race: while the backend is still importing,
      // the Vite proxy answers ECONNREFUSED, and treating that as
      // "signed out" painted the login screen over a valid session
      // (which then pulled the user through a needless re-auth).
      // `loading` stays true through the retries, so the shell keeps
      // showing its boot spinner instead.
      const s = await retryBoot(() => api.auth.status());
      setState({ ...s, loading: false });
    } catch {
      setState({
        logged_in: false,
        username: null,
        avatar: null,
        user_id: null,
        loading: false,
      });
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const logout = useCallback(async () => {
    // Refresh unconditionally. If the server-side logout call fails
    // (network, 401), we still want to re-read the status so the UI
    // reflects whatever truth the server actually has instead of
    // leaving the caller with an unhandled rejection.
    try {
      await api.auth.logout();
    } catch {
      /* swallow — refresh() below will surface the real state */
    } finally {
      // Wipe the SWR cache so a subsequent login doesn't render
      // the previous user's pages from memory before fresh data
      // arrives.
      clearApiCache();
      await refresh();
    }
  }, [refresh]);

  return { ...state, refresh, logout };
}
