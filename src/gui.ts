import GUI from 'lil-gui';
import {
  DEFAULT_INFERENCE_RESOLUTION,
  INFERENCE_RESOLUTIONS,
  type InferenceResolution,
} from './depth/capture.ts';
import {
  availableDevices,
  defaultDevice,
  type DepthDevice,
} from './depth/model.ts';
import {
  defaultRelightingSettings,
  type RelightSettings,
  type ViewMode,
} from './renderer.ts';

export type FacingMode = 'front' | 'back';

export interface CameraResolutionOption {
  readonly width: number;
  readonly height: number;
}

export interface GuiCallbacks {
  readonly onSwitchSource: () => void;
  readonly onSettingsChange: (partial: Partial<RelightSettings>) => void;
  readonly onFacingChange: (facing: FacingMode) => void;
  readonly onAutoOrbitChange: (enabled: boolean) => void;
  readonly onDepthResolutionChange: (resolution: InferenceResolution) => void;
  /** Engine (ONNX execution provider) switch; reloads the depth model on the new one. */
  readonly onEngineChange: (device: DepthDevice) => void;
  readonly onLockToDepthChange: (enabled: boolean) => void;
  readonly onCameraResolutionChange: (
    resolution: CameraResolutionOption,
  ) => void;
}

export interface GuiHandles {
  readonly gui: GUI;
  /** Updates the read-only depth-inference stats row (e.g. "42 ms · 24 fps"). */
  readonly setDepthStats: (text: string) => void;
  /** Updates the read-only render stats row (e.g. "120 fps · 2.7 ms gpu"). */
  readonly setRenderStats: (text: string) => void;
  /** Updates the read-only backend row with what the backend actually loaded. */
  readonly setBackend: (text: string) => void;
  /** Restores the engine dropdown after a failed switch (the reload threw). */
  readonly setEngine: (device: DepthDevice) => void;
}

const VIEW_MODES: readonly ViewMode[] = ['relit', 'camera', 'depth', 'normals'];
const FACING_MODES: readonly FacingMode[] = ['front', 'back'];

/** Only engines this browser exposes are offered; see availableDevices(). */
const ENGINES: readonly DepthDevice[] = availableDevices();

const CAMERA_RESOLUTIONS: Record<string, CameraResolutionOption> = {
  '480p': { width: 640, height: 480 },
  '720p': { width: 1280, height: 720 },
  '1080p': { width: 1920, height: 1080 },
};
export const DEFAULT_CAMERA_RESOLUTION_LABEL = '720p';

interface GuiState {
  intensity: number;
  ambient: number;
  relief: number;
  shadow: number;
  occlusion: number;
  'bulb size': number;
  lightColor: [number, number, number];
  'auto orbit': boolean;
  view: ViewMode;
  camera: FacingMode;
  'camera res': string;
  'depth res': InferenceResolution;
  'lock to depth': boolean;
  engine: DepthDevice;
  backend: string;
  depth: string;
  render: string;
  'switch model / source': () => void;
}

/** The original example's `defineControls` block, rebuilt with lil-gui. */
export function createGui(callbacks: GuiCallbacks): GuiHandles {
  const gui = new GUI({ title: 'controls' });

  const state: GuiState = {
    intensity: defaultRelightingSettings.intensity,
    ambient: defaultRelightingSettings.exposure,
    relief: defaultRelightingSettings.relief,
    shadow: defaultRelightingSettings.shadow,
    occlusion: defaultRelightingSettings.occlusion,
    'bulb size': defaultRelightingSettings.bulbSize,
    lightColor: [...defaultRelightingSettings.lightColor] as [
      number,
      number,
      number,
    ],
    'auto orbit': false,
    view: defaultRelightingSettings.mode,
    camera: 'front',
    'camera res': DEFAULT_CAMERA_RESOLUTION_LABEL,
    'depth res': DEFAULT_INFERENCE_RESOLUTION,
    'lock to depth': false,
    engine: defaultDevice(),
    backend: '—',
    depth: '—',
    render: '—',
    'switch model / source': () => callbacks.onSwitchSource(),
  };

  gui.add(state, 'switch model / source');

  gui
    .add(state, 'intensity', 0, 3.5, 0.05)
    .onChange((value: number) =>
      callbacks.onSettingsChange({ intensity: value }),
    );
  gui
    .add(state, 'ambient', 0, 1.2, 0.05)
    .onChange((value: number) =>
      callbacks.onSettingsChange({ exposure: value }),
    );
  gui
    .add(state, 'relief', 0, 2.5, 0.05)
    .onChange((value: number) => callbacks.onSettingsChange({ relief: value }));
  gui
    .add(state, 'shadow', 0, 1, 0.05)
    .onChange((value: number) => callbacks.onSettingsChange({ shadow: value }));
  gui
    .add(state, 'occlusion', 0, 1, 0.05)
    .onChange((value: number) =>
      callbacks.onSettingsChange({ occlusion: value }),
    );
  gui
    .add(state, 'bulb size', 0.01, 0.1, 0.005)
    .onChange((value: number) =>
      callbacks.onSettingsChange({ bulbSize: value }),
    );
  gui
    .addColor(state, 'lightColor', 1)
    .onChange((value: [number, number, number]) =>
      callbacks.onSettingsChange({ lightColor: [...value] }),
    );
  gui
    .add(state, 'auto orbit')
    .onChange((value: boolean) => callbacks.onAutoOrbitChange(value));
  gui
    .add(state, 'view', [...VIEW_MODES])
    .onChange((value: ViewMode) => callbacks.onSettingsChange({ mode: value }));
  gui
    .add(state, 'camera', [...FACING_MODES])
    .onChange((value: FacingMode) => callbacks.onFacingChange(value));
  gui
    .add(state, 'camera res', Object.keys(CAMERA_RESOLUTIONS))
    .onChange((value: string) => {
      const resolution = CAMERA_RESOLUTIONS[value];
      if (resolution) {
        callbacks.onCameraResolutionChange(resolution);
      }
    });
  gui
    .add(state, 'depth res', [...INFERENCE_RESOLUTIONS])
    .onChange((value: InferenceResolution) =>
      callbacks.onDepthResolutionChange(value),
    );
  gui
    .add(state, 'lock to depth')
    .onChange((value: boolean) => callbacks.onLockToDepthChange(value));

  const engineControl = gui
    .add(state, 'engine', [...ENGINES])
    .onChange((value: DepthDevice) => callbacks.onEngineChange(value));

  const backendRow = gui.add(state, 'backend').disable();
  const depthStats = gui.add(state, 'depth').disable();
  const renderStats = gui.add(state, 'render').disable();

  return {
    gui,
    setDepthStats: (text: string) => {
      state.depth = text;
      depthStats.updateDisplay();
    },
    setRenderStats: (text: string) => {
      state.render = text;
      renderStats.updateDisplay();
    },
    setBackend: (text: string) => {
      state.backend = text;
      backendRow.updateDisplay();
    },
    setEngine: (device: DepthDevice) => {
      state.engine = device;
      engineControl.updateDisplay();
    },
  };
}
