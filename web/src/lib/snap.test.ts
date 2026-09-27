import { describe, expect, it } from 'vitest';
import { snappedLabel } from './snap';

describe('snappedLabel', () => {
  it('names the thread only for an exact standard clearance size', () => {
    expect(snappedLabel(3.4)).toBe('M3');
    expect(snappedLabel(4.5)).toBe('M4');
    expect(snappedLabel(5.5)).toBe('M5');
    expect(snappedLabel(6.6)).toBe('M6');
    expect(snappedLabel(9)).toBe('M8');
  });
  it('says "standard size" for anything else', () => {
    expect(snappedLabel(5.4)).toBe('standard size');
    expect(snappedLabel(10)).toBe('standard size');
    expect(snappedLabel(undefined)).toBe('standard size');
  });
});
