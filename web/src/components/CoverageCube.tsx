import type { CSSProperties } from 'react';
import type { Face } from '../api/types';
import type { CaptureItem } from '../state/store';

const MONO = "'Geist Mono', monospace";

/** Net layout: a cross with FRONT in the middle row, matching an unfolded cube. */
const ORDER: Face[] = ['top', 'left', 'front', 'right', 'back', 'bottom'];
const LABEL: Record<Face, string> = { front: 'FRONT', back: 'BACK', left: 'LEFT', right: 'RIGHT', top: 'TOP', bottom: 'BOTTOM' };

/** Face cell styles, ported verbatim from the `F` map in Analyzing v2.dc.html (~line 455). Only
 * `empty` and `photo` apply here: Capture only knows what was photographed, never what the AI
 * will draw or mirror later. */
const F = {
  empty: { bs: 'solid', bc: 'var(--line)', bg: 'var(--inset)', tc: 'var(--muted)' },
  photo: { bs: 'solid', bc: 'var(--trusted)', bg: 'color-mix(in oklch, var(--trusted) 18%, var(--surface))', tc: 'var(--trusted)' },
} as const;

export interface CoverageCubeProps {
  items: CaptureItem[];
}

/** Unfolded net of the 6 faces, filled from the items' face tags. `auto` tags do not count. */
export function CoverageCube({ items }: CoverageCubeProps) {
  const counts: Record<Face, number> = { front: 0, back: 0, left: 0, right: 0, top: 0, bottom: 0 };
  for (const it of items) {
    if (it.face !== 'auto') counts[it.face] += 1;
  }
  const covered = Object.values(counts).filter((n) => n > 0).length;

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 12 }}>
        <span style={{ fontSize: 15, fontWeight: 600 }}>Coverage</span>
        <span style={{ fontSize: 13, color: 'var(--muted)', whiteSpace: 'nowrap' }}>
          <span style={{ fontFamily: MONO, fontSize: 14, color: 'var(--ink)' }}>{covered}</span> of 6 faces
        </span>
      </div>
      <div
        style={{
          display: 'grid',
          gridTemplateAreas: '". top . ." "left front right back" ". bottom . ."',
          gridTemplateColumns: 'repeat(4, minmax(0, 1fr))',
          gap: 8,
        }}
      >
        {ORDER.map((face) => {
          const n = counts[face];
          const f = n > 0 ? F.photo : F.empty;
          const cell: CSSProperties = {
            gridArea: face, boxSizing: 'border-box', aspectRatio: '1', border: `1.5px ${f.bs} ${f.bc}`,
            background: f.bg, borderRadius: 6, padding: '6px 7px', display: 'flex', flexDirection: 'column',
            justifyContent: 'space-between',
          };
          return (
            <div key={face} style={cell}>
              <span style={{ fontFamily: MONO, fontSize: 10, letterSpacing: '0.05em', color: 'var(--muted)' }}>{LABEL[face]}</span>
              <span style={{ fontFamily: MONO, fontSize: 12, fontWeight: 600, color: f.tc }}>{n > 0 ? `✓ ×${n}` : '—'}</span>
            </div>
          );
        })}
      </div>
      <div style={{ display: 'flex', gap: 16, marginTop: 12, fontSize: 12, color: 'var(--muted)' }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ width: 14, height: 10, border: '1.5px solid var(--trusted)', background: 'color-mix(in oklch, var(--trusted) 18%, transparent)', borderRadius: 2 }} />
          Photographed
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ width: 14, height: 10, border: '1.5px solid var(--line)', background: 'var(--inset)', borderRadius: 2 }} />
          Empty
        </span>
      </div>
    </div>
  );
}
