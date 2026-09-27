import { createContext, useCallback, useContext, useMemo, useReducer, useRef, type Dispatch, type ReactNode } from 'react';
import type { AiSettings, Analysis, Face, GeometrySettings, Job, ModelResult } from '../api/types';
import { resetDeadline } from '../lib/deadline';

export type Screen = 'capture' | 'analyzing' | 'review' | 'model';
export type CaptureKind = 'auto' | 'sketch' | 'photo' | 'drawing';
export interface CaptureItem { id: string; file: File; url: string; face: Face | 'auto'; kind: CaptureKind }

export interface State {
  screen: Screen;
  items: CaptureItem[];
  reference: string;
  ai: AiSettings;
  jobId: string | null;
  job: Job | null;
  analysis: Analysis | null;
  typed: Record<string, number>;
  rejected: Face[];
  geometry: GeometrySettings;
  model: ModelResult | null;
}

export type Action =
  | { type: 'ADD_FILES'; items: CaptureItem[] }
  | { type: 'SET_ITEM'; id: string; patch: Partial<Omit<CaptureItem, 'id'>> }
  | { type: 'REMOVE_ITEM'; id: string }
  | { type: 'SET_REFERENCE'; reference: string }
  | { type: 'SET_AI'; patch: Partial<AiSettings> }
  | { type: 'START_JOB'; jobId: string }
  | { type: 'JOB_UPDATE'; job: Job }
  | { type: 'ANALYSIS'; analysis: Analysis }
  | { type: 'TYPE_VALUE'; path: string; value: number }
  | { type: 'TOGGLE_REJECT'; face: Face }
  | { type: 'SET_GEOMETRY'; patch: Partial<GeometrySettings> }
  | { type: 'MODEL'; model: ModelResult | null }
  | { type: 'GOTO'; screen: Screen }
  | { type: 'RESET' };

export const MAX_ITEMS = 6;

export const initialAi: AiSettings = {
  use_reader: true, use_qwen_image: true, use_rescue: true, use_triposr: true, use_solaria: true,
  seed: 7, randomize_seed: false, attempts: 2,
};

export const initialGeometry: GeometrySettings = {
  snap: true, clearance: 'medium', finish: 'none', finish_mm: 1.0, finish_edges: 'all_vertical',
};

export const initialState: State = {
  screen: 'capture', items: [], reference: '', ai: initialAi, jobId: null, job: null, analysis: null,
  typed: {}, rejected: [], geometry: initialGeometry, model: null,
};

let seq = 0;
/** Build a CaptureItem (with an object URL for previews) to pass to ADD_FILES. */
export function toCaptureItem(file: File, face: Face | 'auto' = 'auto', kind: CaptureKind = 'auto'): CaptureItem {
  seq += 1;
  return { id: `img-${Date.now().toString(36)}-${seq}`, file, url: URL.createObjectURL(file), face, kind };
}

export function reducer(state: State, action: Action): State {
  switch (action.type) {
    case 'ADD_FILES':
      return { ...state, items: [...state.items, ...action.items].slice(0, MAX_ITEMS) };
    case 'SET_ITEM':
      return { ...state, items: state.items.map((it) => (it.id === action.id ? { ...it, ...action.patch } : it)) };
    case 'REMOVE_ITEM':
      return { ...state, items: state.items.filter((it) => it.id !== action.id) };
    case 'SET_REFERENCE':
      return { ...state, reference: action.reference };
    case 'SET_AI':
      return { ...state, ai: { ...state.ai, ...action.patch } };
    case 'START_JOB':
      return { ...state, screen: 'analyzing', jobId: action.jobId, job: null, analysis: null, typed: {}, rejected: [], model: null };
    case 'JOB_UPDATE':
      return action.job.job_id === state.jobId ? { ...state, job: action.job } : state;
    case 'ANALYSIS':
      return { ...state, analysis: action.analysis };
    case 'TYPE_VALUE':
      return { ...state, typed: { ...state.typed, [action.path]: action.value } };
    case 'TOGGLE_REJECT':
      return {
        ...state,
        rejected: state.rejected.includes(action.face) ? state.rejected.filter((f) => f !== action.face) : [...state.rejected, action.face],
      };
    case 'SET_GEOMETRY':
      return { ...state, geometry: { ...state.geometry, ...action.patch } };
    case 'MODEL':
      return { ...state, model: action.model };
    case 'GOTO':
      return { ...state, screen: action.screen };
    case 'RESET':
      return { ...initialState, ai: state.ai, geometry: state.geometry };
    default:
      return state;
  }
}

interface StoreValue { state: State; dispatch: Dispatch<Action> }
const StoreContext = createContext<StoreValue | null>(null);

export function StoreProvider({ children, initial }: { children: ReactNode; initial?: Partial<State> }) {
  const [state, rawDispatch] = useReducer(reducer, { ...initialState, ...initial });
  const stateRef = useRef(state);
  stateRef.current = state;
  const dispatch = useCallback<Dispatch<Action>>((action) => {
    // Side effects live here, not in the reducer (StrictMode runs reducers twice).
    if (action.type === 'START_JOB') resetDeadline();
    if (action.type === 'REMOVE_ITEM') {
      const it = stateRef.current.items.find((i) => i.id === action.id);
      if (it) URL.revokeObjectURL(it.url);
    }
    rawDispatch(action);
  }, []);
  const value = useMemo(() => ({ state, dispatch }), [state, dispatch]);
  return <StoreContext.Provider value={value}>{children}</StoreContext.Provider>;
}

export function useStore(): StoreValue {
  const v = useContext(StoreContext);
  if (!v) throw new Error('useStore must be used inside StoreProvider');
  return v;
}
