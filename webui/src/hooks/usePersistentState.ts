import { useEffect, useRef, useState } from "react";

/**
 * useState + localStorage. Same signature as useState (initial value only).
 *
 * - Reads once on mount; writes on every change (debounced via microtask coalescing
 *   is unnecessary at this volume — JSON of small forms is sub-millisecond).
 * - If JSON parse fails, falls back to `initial` and overwrites the slot.
 * - Schema-tolerant: if you add new fields to `initial` later, they will be
 *   merged in (only when `initial` is a plain object) so old saves don't break.
 */
export function usePersistentState<T>(
  key: string,
  initial: T,
): [T, React.Dispatch<React.SetStateAction<T>>] {
  const storageKey = `archaea.${key}`;
  const [value, setValue] = useState<T>(() => {
    if (typeof window === "undefined") return initial;
    try {
      const raw = window.localStorage.getItem(storageKey);
      if (raw == null) return initial;
      const parsed = JSON.parse(raw) as T;
      if (
        initial != null &&
        typeof initial === "object" &&
        !Array.isArray(initial) &&
        parsed != null &&
        typeof parsed === "object" &&
        !Array.isArray(parsed)
      ) {
        return { ...(initial as object), ...(parsed as object) } as T;
      }
      return parsed;
    } catch {
      return initial;
    }
  });

  const first = useRef(true);
  useEffect(() => {
    if (first.current) {
      first.current = false;
      return;
    }
    try {
      window.localStorage.setItem(storageKey, JSON.stringify(value));
    } catch {
      /* quota / privacy mode — silently ignore */
    }
  }, [storageKey, value]);

  return [value, setValue];
}
