import { useStore } from '../state/store';

/** Placeholder; replaced by the Review screen task. */
export function Review() {
  const { state } = useStore();
  return (
    <section data-screen="Review" style={{ padding: '32px 40px' }}>
      <h1 style={{ margin: 0, fontSize: 34, fontWeight: 600, letterSpacing: '-0.02em' }}>Review</h1>
      <p style={{ color: 'var(--muted)' }}>Screen: {state.screen}</p>
    </section>
  );
}
