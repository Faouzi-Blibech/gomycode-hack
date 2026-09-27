// Clearance holes the snapper produces (medium fit), by thread. Only an exact match names a thread.
const THREADS: [number, string][] = [[3.4, 'M3'], [4.5, 'M4'], [5.5, 'M5'], [6.6, 'M6'], [9, 'M8']];

/** Label for a snapped value: "M5" when it is exactly a standard clearance hole, otherwise "standard size". */
export function snappedLabel(value: number | undefined): string {
  if (typeof value !== 'number') return 'standard size';
  return THREADS.find(([d]) => Math.abs(d - value) < 1e-9)?.[1] ?? 'standard size';
}
