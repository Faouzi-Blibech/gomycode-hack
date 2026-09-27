import { useStore } from '../state/store';

/** Placeholder; replaced by the Capture screen task. */
export function Capture() {
  const { state } = useStore();
  return (
    <section data-screen="Capture" style={{ padding: '32px 40px' }}>
      <h1 style={{ margin: 0, fontSize: 34, fontWeight: 600, letterSpacing: '-0.02em' }}>Capture</h1>
      <p style={{ color: 'var(--muted)' }}>Screen: {state.screen}</p>
    </section>
  );
}
