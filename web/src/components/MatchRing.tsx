import type { Face } from '../api/types';

export const IOU_GREEN = 0.85;
const FACES: Face[] = ['front', 'back', 'left', 'right', 'top', 'bottom'];
// The views are white-part-on-black masks. In the light theme the trace tokens (no filter, multiply) would show
// black blocks, so invert them there: a dark part on the light trace background.
const CSS = `.s2c-mini{filter:var(--trace-filter);mix-blend-mode:var(--trace-blend)}
:root[data-theme=light] .s2c-mini{filter:invert(1) contrast(0.9)}`;

export interface MatchRingProps {
  iouMean: number | null;
  iou: Partial<Record<Face, number>>;
  views: Partial<Record<Face, string>>;
}

/**
 * Round-trip match: the built part's silhouettes against the input. The ring uses the same conic-gradient
 * construction as the shell timer; the mini views use the design's trace tokens so they read in both themes.
 */
export function MatchRing({ iouMean, iou, views }: MatchRingProps) {
  const has = typeof iouMean === 'number';
  const good = has && iouMean >= IOU_GREEN;
  const color = !has ? 'var(--muted)' : good ? 'var(--trusted)' : 'var(--check)';
  const deg = has ? Math.max(0, Math.min(1, iouMean)) * 360 : 0;
  const faces = FACES.filter((f) => views[f]);
  return (
    <div style={{ width: 212, padding: '12px 14px', borderRadius: 12, background: 'var(--raised)', boxShadow: 'var(--shadow)', display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <div role="img" aria-label={has ? `Match ${Math.round(iouMean * 100)} percent` : 'No match score'}
          style={{ width: 52, height: 52, flex: 'none', borderRadius: '50%', background: `conic-gradient(${color} ${deg}deg, var(--line) 0)`, display: 'grid', placeItems: 'center' }}>
          <div style={{ width: 42, height: 42, borderRadius: '50%', background: 'var(--raised)', display: 'grid', placeItems: 'center', fontFamily: "'Geist Mono', monospace", fontSize: 14, fontWeight: 500, fontVariantNumeric: 'tabular-nums' }}>
            <span>{has ? Math.round(iouMean * 100) : '—'}{has && <span style={{ fontSize: 10, fontWeight: 300, color: 'var(--muted)' }}>%</span>}</span>
          </div>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 2, minWidth: 0 }}>
          <span style={{ fontSize: 12, color: 'var(--muted)' }}>Match with your input</span>
          <span style={{ fontSize: 13, fontWeight: 500, color }}>
            {!has ? 'No outline to compare' : good ? '✓ Shapes line up' : '! Check the dimensions'}
          </span>
        </div>
      </div>
      {faces.length > 0 && <style>{CSS}</style>}
      {faces.length > 0 && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0,1fr))', gap: 6 }}>
          {faces.map((f) => {
            const v = iou[f];
            const low = typeof v === 'number' && v < IOU_GREEN;
            return (
              <figure key={f} style={{ margin: 0, display: 'flex', flexDirection: 'column', gap: 3 }}>
                <div style={{ aspectRatio: '1', borderRadius: 7, overflow: 'hidden', background: 'var(--trace-bg)', border: `1px solid ${low ? 'var(--check)' : 'var(--line)'}` }}>
                  <img className="s2c-mini" src={views[f]} alt={`${f} view of the built part`} loading="lazy"
                    style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block' }} />
                </div>
                <figcaption style={{ fontFamily: "'Geist Mono', monospace", fontSize: 10, color: low ? 'var(--check)' : 'var(--muted)', whiteSpace: 'nowrap', textAlign: 'center' }}>
                  {f}{typeof v === 'number' ? ` · ${v.toFixed(2)}` : ''}
                </figcaption>
              </figure>
            );
          })}
        </div>
      )}
    </div>
  );
}
