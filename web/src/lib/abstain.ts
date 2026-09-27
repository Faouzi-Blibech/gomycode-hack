import type { Abstain } from '../api/types';

/** Abstains the user fixes by typing a number (Review); anything else (label, outline) needs new images. */
export function isDimensionAbstain(a: Abstain): boolean {
  return a.stage === 'dimensions' || a.stage === 'build' || (a.missing?.length ?? 0) > 0;
}

/** The Analyzing screen's call to action for a stopped analysis. */
export function analyzingCta(a: Abstain): { label: string; to: 'review' | 'capture' } {
  if (!isDimensionAbstain(a)) return { label: 'Back to capture →', to: 'capture' };
  const n = a.missing?.length ?? 0;
  return { label: n ? `Fix ${n} ${n === 1 ? 'value' : 'values'} →` : 'Fix a value →', to: 'review' };
}

/** Shown when the server no longer knows the analysis (404): its files were swept after one hour. */
export const EXPIRED = {
  title: 'This analysis expired',
  remedy: 'Files are deleted after one hour. Analyze your images again.',
  action: 'Analyze again',
} as const;
