import { Shell, type NavItem } from './components/Shell';
import { countChecks } from './lib/provenance';
import { Analyzing } from './screens/Analyzing';
import { Capture } from './screens/Capture';
import { Model } from './screens/Model';
import { Review } from './screens/Review';
import { useStore, type Screen, type State } from './state/store';

export function buildNav(state: State): NavItem[] {
  const { screen, items, job, analysis, model, typed } = state;
  const running = screen === 'analyzing' && (!job || job.status === 'running');
  const inCapture = screen === 'capture' || screen === 'analyzing';
  const spec = analysis?.spec ?? null;
  const stop = !!analysis?.abstain;
  const nCheck = spec ? countChecks(spec, typed) : 0;
  const n = items.length;

  const capture: NavItem = {
    num: '01', label: 'Capture', screen: 'capture',
    st: inCapture ? 'current' : 'done',
    sub: screen === 'analyzing'
      ? (running ? 'Analyzing…' : job?.status === 'done' ? 'Analysis done' : job?.status === 'failed' ? 'Analysis failed' : 'Stopped')
      : n ? `${n} ${n === 1 ? 'image' : 'images'}` : 'Add your sketches',
    subC: screen === 'analyzing' ? (job?.status === 'failed' ? 'var(--stop)' : 'var(--accent-ink)') : null,
    spinning: running,
  };
  const review: NavItem = {
    num: '02', label: 'Review', screen: 'review',
    st: !analysis ? 'locked' : screen === 'review' ? 'current' : screen === 'model' ? 'done' : 'open',
    sub: !analysis ? 'After analysis' : stop ? 'Needs your input' : nCheck ? `${nCheck} to check` : 'All checked',
    subC: !analysis ? null : stop ? 'var(--stop)' : nCheck ? 'var(--check)' : 'var(--trusted)',
  };
  const modelOpen = !!spec && !stop && (!!model || screen === 'model');
  const modelItem: NavItem = {
    num: '03', label: 'Model', screen: 'model',
    st: !modelOpen ? 'locked' : screen === 'model' ? 'current' : 'open',
    sub: !modelOpen ? (spec && !stop ? 'Build to open' : 'After review') : model ? 'Built' : 'Building…',
  };
  return [capture, review, modelItem];
}

export function App() {
  const { state, dispatch } = useStore();
  const onNav = (screen: Screen) => {
    // Going back to Capture while a finished analysis exists keeps it; Analyzing is reached through Capture.
    if (screen === 'capture' && state.screen === 'analyzing') return;
    dispatch({ type: 'GOTO', screen });
  };
  return (
    <Shell nav={buildNav(state)} onNav={onNav}>
      {state.screen === 'capture' && <Capture />}
      {state.screen === 'analyzing' && <Analyzing />}
      {state.screen === 'review' && <Review />}
      {state.screen === 'model' && <Model />}
    </Shell>
  );
}
