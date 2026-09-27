import { describe, expect, it } from 'vitest';
import { listPhrase } from './words';

describe('listPhrase', () => {
  it('joins words the way a person writes them', () => {
    expect(listPhrase([])).toBe('');
    expect(listPhrase(['width'])).toBe('width');
    expect(listPhrase(['width', 'depth'])).toBe('width and depth');
    expect(listPhrase(['width', 'height', 'depth'])).toBe('width, height and depth');
  });
});
