import { useStore } from '../state/store';

/** Placeholder; replaced by the Model screen task. */
export function Model() {
  const { state } = useStore();
  return (
    <section data-screen="Model" style={{ padding: '32px 40px' }}>
      <h1 style={{ margin: 0, fontSize: 34, fontWeight: 600, letterSpacing: '-0.02em' }}>Model</h1>
      <p style={{ color: 'var(--muted)' }}>Screen: {state.screen}</p>
    </section>
  );
}
