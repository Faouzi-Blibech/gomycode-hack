import { useEffect, useState } from 'react';

export const DEADLINE_KEY = 's2c_deadline';
export const TTL_MS = 60 * 60 * 1000;

function read(): number {
  let d = 0;
  try { d = Number(localStorage.getItem(DEADLINE_KEY)) || 0; } catch { /* storage blocked */ }
  if (!d || d < Date.now()) {
    d = Date.now() + TTL_MS;
    try { localStorage.setItem(DEADLINE_KEY, String(d)); } catch { /* storage blocked */ }
  }
  return d;
}

const listeners = new Set<(d: number) => void>();

/** Restart the 60-minute privacy countdown. Call when a new analysis starts. */
export function resetDeadline(): number {
  const d = Date.now() + TTL_MS;
  try { localStorage.setItem(DEADLINE_KEY, String(d)); } catch { /* storage blocked */ }
  listeners.forEach((l) => l(d));
  return d;
}

/** Returns the deadline (epoch ms) and the remaining seconds, ticking once a second. */
export function useDeadline(): { deadline: number; remaining: number } {
  const [deadline, setDeadline] = useState(read);
  const [now, setNow] = useState(Date.now);
  useEffect(() => {
    listeners.add(setDeadline);
    const t = window.setInterval(() => setNow(Date.now()), 1000);
    return () => { listeners.delete(setDeadline); window.clearInterval(t); };
  }, []);
  return { deadline, remaining: Math.max(0, deadline - now) / 1000 };
}
