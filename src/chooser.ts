import { ModelSize, MODEL_SIZES } from './depth/model.ts';

/**
 * Startup overlay: pick a source (camera / demo photo / uploaded photo), a depth model
 * size, and start. There is no download-cache toggle as in the original; transformers.js
 * manages its own model download and browser cache.
 */
export const SourceChoice = {
  CAMERA: 'camera',
  DEMO: 'demo',
  UPLOAD: 'upload',
} as const;
export type SourceChoice = (typeof SourceChoice)[keyof typeof SourceChoice];

const SOURCE_LABELS: [SourceChoice, string][] = [
  [SourceChoice.CAMERA, 'live camera'],
  [SourceChoice.DEMO, 'demo photo'],
  [SourceChoice.UPLOAD, 'your photo…'],
];

const MODEL_LABELS: Record<ModelSize, string> = {
  small: 'small · fastest',
  base: 'base',
  large: 'large · best quality',
};

export class SourceChooser {
  source: SourceChoice = SourceChoice.CAMERA;
  model: ModelSize = ModelSize.SMALL;
  uploadedImage: ImageBitmap | undefined;

  readonly #panel = document.querySelector('.chooser') as HTMLDivElement;
  readonly #sourceRow = document.querySelector('.source-row') as HTMLDivElement;
  readonly #modelRow = document.querySelector('.model-row') as HTMLDivElement;
  readonly #error = document.querySelector(
    '.chooser-error',
  ) as HTMLParagraphElement;
  readonly #photoInput = document.querySelector(
    '.photo-input',
  ) as HTMLInputElement;
  readonly #signal: AbortSignal;

  constructor(onStart: () => void, signal: AbortSignal) {
    this.#signal = signal;
    this.#buildSourceRow();
    this.#buildModelRow();
    const startButton = document.querySelector(
      '.start-button',
    ) as HTMLButtonElement;
    startButton.addEventListener('click', onStart, { signal });
  }

  show(errorText?: string): void {
    this.#error.textContent = errorText ?? '';
    this.#error.hidden = !errorText;
    this.#markSelected();
    this.#panel.hidden = false;
  }

  hide(): void {
    this.#panel.hidden = true;
  }

  destroy(): void {
    this.uploadedImage?.close();
  }

  #buildSourceRow(): void {
    for (const [choice, label] of SOURCE_LABELS) {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = label;
      button.dataset.choice = choice;
      button.addEventListener('click', () => this.#selectSource(choice), {
        signal: this.#signal,
      });
      this.#sourceRow.append(button);
    }
    this.#photoInput.addEventListener('change', () => this.#takeUpload(), {
      signal: this.#signal,
    });
  }

  #buildModelRow(): void {
    for (const size of MODEL_SIZES) {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = MODEL_LABELS[size];
      button.dataset.model = size;
      button.addEventListener(
        'click',
        () => {
          this.model = size;
          this.#markSelected();
        },
        { signal: this.#signal },
      );
      this.#modelRow.append(button);
    }
  }

  #selectSource(choice: SourceChoice): void {
    if (choice === SourceChoice.UPLOAD) {
      this.#photoInput.click();
      return;
    }
    this.source = choice;
    this.#markSelected();
  }

  #takeUpload(): void {
    const file = this.#photoInput.files?.[0];
    if (!file) {
      return;
    }
    void createImageBitmap(file).then((bitmap) => {
      this.uploadedImage?.close();
      this.uploadedImage = bitmap;
      this.source = SourceChoice.UPLOAD;
      this.#markSelected();
    });
  }

  #markSelected(): void {
    for (const button of this.#sourceRow.querySelectorAll('button')) {
      button.classList.toggle(
        'selected',
        button.dataset.choice === this.source,
      );
    }
    for (const button of this.#modelRow.querySelectorAll('button')) {
      button.classList.toggle('selected', button.dataset.model === this.model);
    }
  }
}
