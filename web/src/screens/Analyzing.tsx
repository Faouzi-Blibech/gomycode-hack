import { useStore } from '../state/store';

/** Placeholder; replaced by the Analyzing screen task. */
export function Analyzing() {
  const { state } = useStore();
  return (
    <section data-screen="Analyzing" style={{ padding: '32px 40px' }}>
      <h1 style={{ margin: 0, fontSize: 34, fontWeight: 600, letterSpacing: '-0.02em' }}>Analyzing</h1>
      <p style={{ color: 'var(--muted)' }}>Screen: {state.screen}</p>
    </section>
  );
}
