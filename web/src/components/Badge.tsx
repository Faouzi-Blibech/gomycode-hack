import type { CSSProperties } from 'react';
import type { Provenance } from '../api/types';
import { BADGE } from '../lib/provenance';

export interface BadgeProps { prov: Provenance | 'required'; onClick?: () => void; style?: CSSProperties }

/** Provenance pill, styled like the badges in Review v2.dc.html. */
export function Badge({ prov, onClick, style }: BadgeProps) {
  const b = BADGE[prov] ?? BADGE.default;
  const css: CSSProperties = {
    display: 'inline-flex', alignItems: 'center', gap: 6, height: 26, padding: '0 10px 0 8px', borderRadius: 13,
    border: `1.5px ${b.line} ${b.fg}`, background: b.bg, color: b.fg, font: 'inherit', fontSize: 12, fontWeight: 500,
    cursor: onClick ? 'pointer' : 'default', whiteSpace: 'nowrap', boxSizing: 'border-box', ...style,
  };
  const inner = (<><span aria-hidden="true">{b.icon}</span>{b.label}</>);
  return onClick
    ? <button type="button" onClick={onClick} style={css}>{inner}</button>
    : <span style={css}>{inner}</span>;
}
