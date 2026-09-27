import type { Analysis, PartRequest, PartType, Spec } from '../api/types';

export const PART_LABELS: Record<PartType, string> = { plate: 'Flat plate', l_bracket: 'L-bracket', spacer: 'Spacer', flange: 'Flange' };

/** The values each part type needs, in the order the card lists them (see the Contract in the plan). */
export const REQUIRED: Record<PartType, string[]> = {
  plate: ['width_mm', 'height_mm', 'thickness_mm'],
  l_bracket: ['leg_a_mm', 'leg_b_mm', 'width_mm', 'thickness_mm'],
  spacer: ['outer_diameter_mm', 'inner_diameter_mm', 'length_mm'],
  flange: ['outer_diameter_mm', 'inner_diameter_mm', 'thickness_mm', 'bolt_circle_diameter_mm', 'bolt_hole_diameter_mm', 'bolt_count'],
};

const NAMES: Record<string, string> = {
  width_mm: 'Width', height_mm: 'Height', thickness_mm: 'Thickness', leg_a_mm: 'Horizontal leg', leg_b_mm: 'Vertical leg',
  outer_diameter_mm: 'Outer diameter', inner_diameter_mm: 'Inner diameter', length_mm: 'Length',
  bolt_circle_diameter_mm: 'Bolt circle', bolt_hole_diameter_mm: 'Bolt hole', bolt_count: 'Bolt count',
};

export function valueLabel(key: string): string {
  if (NAMES[key]) return NAMES[key];
  const s = key.replace(/_mm$/, '').replace(/_/g, ' ');
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export function formatValue(key: string, v: number): string {
  const n = String(+v.toFixed(2));
  return key.endsWith('_mm') ? `${n} mm` : n;
}

export interface PartRow { key: string; label: string; value: number | null }

/** Every value of the part: the ones the user wrote, then the missing ones (value null). No duplicates. */
export function partRows(part: PartRequest | null, missing: string[]): PartRow[] {
  const rows: PartRow[] = [];
  const seen = new Set<string>();
  const add = (key: string, value: number | null) => {
    if (seen.has(key)) return;
    seen.add(key);
    rows.push({ key, label: valueLabel(key), value });
  };
  const values = part?.values ?? {};
  const order = part ? REQUIRED[part.type] ?? [] : [];
  for (const k of order) if (typeof values[k] === 'number' && !missing.includes(k)) add(k, values[k]);
  for (const [k, v] of Object.entries(values)) if (typeof v === 'number' && !missing.includes(k)) add(k, v);
  for (const k of missing) add(k, null);
  return rows;
}

/** The analysis the Model screen builds from: no job, so the request id is empty (sent as null). */
export function describedAnalysis(spec: Spec): Analysis {
  return { request_id: '', spec, abstain: null, filled_by: {} };
}

export interface FrontPreview { viewBox: string; path: string; holes: { cx: number; cy: number; r: number }[] }

/** SVG data for the front outline (y flipped so up is up), with its inner loops and front holes. */
export function frontPreview(spec: Spec): FrontPreview | null {
  const front = spec.views?.front;
  if (!front || front.outer.length < 3) return null;
  const pts = [...front.outer, ...front.inner.flat()];
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const w = Math.max(maxX - minX, 1e-6), h = Math.max(maxY - minY, 1e-6);
  const pad = Math.max(w, h) * 0.06;
  const f = (n: number) => String(+n.toFixed(2));
  const loop = (l: [number, number][]) =>
    l.map(([x, y], i) => `${i ? 'L' : 'M'}${f(x - minX)} ${f(maxY - y)}`).join(' ') + ' Z';
  const path = [front.outer, ...front.inner].filter((l) => l.length >= 3).map(loop).join(' ');
  const holes: FrontPreview['holes'] = [];
  for (const ft of spec.features) {
    if (ft.type === 'hole' && ft.face === 'front') {
      holes.push({ cx: +(ft.a_mm - minX).toFixed(2), cy: +(maxY - ft.b_mm).toFixed(2), r: ft.diameter_mm / 2 });
    }
  }
  return { viewBox: `${f(-pad)} ${f(-pad)} ${f(w + 2 * pad)} ${f(h + 2 * pad)}`, path, holes };
}
