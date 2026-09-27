import { describe, expect, it } from 'vitest';
import type { Abstain } from '../api/types';
import { analyzingCta, isDimensionAbstain } from './abstain';

const ab = (p: Partial<Abstain>): Abstain => ({ stage: 'dimensions', reason: 'missing_x', remedy: 'Enter the width in mm.', partial: {}, ...p });

describe('abstain actions', () => {
  it('counts the missing values', () => {
    expect(analyzingCta(ab({ missing: ['envelope.x_mm'] }))).toEqual({ label: 'Fix 1 value →', to: 'review' });
    expect(analyzingCta(ab({ missing: ['envelope.x_mm', 'envelope.y_mm', 'envelope.z_mm'] }))).toEqual({ label: 'Fix 3 values →', to: 'review' });
    expect(analyzingCta(ab({ reason: 'invalid_value', missing: [] }))).toEqual({ label: 'Fix a value →', to: 'review' });
  });
  it('sends label and outline stops back to capture', () => {
    const face = ab({ stage: 'label', reason: 'face_unknown', missing: [] });
    const outline = ab({ stage: 'outline', reason: 'no_outline' });
    expect(isDimensionAbstain(face)).toBe(false);
    expect(analyzingCta(face)).toEqual({ label: 'Back to capture →', to: 'capture' });
    expect(analyzingCta(outline)).toEqual({ label: 'Back to capture →', to: 'capture' });
    expect(isDimensionAbstain(ab({ stage: 'build', reason: 'hole_outside' }))).toBe(true);
  });
});
