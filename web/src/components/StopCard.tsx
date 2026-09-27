export interface StopCardProps { title: string; remedy: string; actionLabel?: string; onAction?: () => void; kicker?: string }

/** Abstain card: shows the remedy verbatim, with at most one action. Layout from the stop card in Review v2.dc.html. */
export function StopCard({ title, remedy, actionLabel, onAction, kicker = '[ STOPPED ]' }: StopCardProps) {
  return (
    <div role="alert" style={{ borderRadius: 14, border: '1.5px dashed var(--stop)', background: 'color-mix(in oklch, var(--stop) 10%, var(--surface))', boxShadow: 'var(--shadow)', padding: '18px 20px', display: 'grid', gridTemplateColumns: 'minmax(0,1fr) auto', gap: '6px 16px', alignItems: 'center' }}>
      <span style={{ gridColumn: '1 / 3', fontFamily: "'Geist Mono', monospace", fontSize: 11, letterSpacing: '0.1em', color: 'var(--stop)' }}>{kicker}</span>
      <div>
        <div style={{ fontSize: 21, fontWeight: 600, letterSpacing: '-0.01em' }}>{title}</div>
        <div style={{ fontSize: 15, marginTop: 4 }}>{remedy}</div>
      </div>
      {actionLabel && onAction && (
        <button type="button" onClick={onAction} style={{ height: 48, padding: '0 20px', borderRadius: 12, border: 'none', background: 'var(--stop)', color: 'var(--raised)', font: 'inherit', fontSize: 15, fontWeight: 600, cursor: 'pointer', whiteSpace: 'nowrap' }}>{actionLabel}</button>
      )}
    </div>
  );
}
