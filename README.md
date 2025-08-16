
# Ripple: Sound Wave to SVG Visualizer

**Ripple** is a Python tool for transforming audio waveforms (WAV files) into beautiful SVG visualizations. It is designed for artists, educators, and anyone interested in exploring the structure of sound through visual art. The project also provides utilities for merging and scaling SVGs, making it easy to create composite visualizations.

## What does Ripple do?

- Converts mono or stereo WAV files into SVG images representing the waveform and its mathematical envelope.
- Supports both linear (waveform) and circular (radial) SVG visualizations.
- Allows customization of the output curve color, smoothness, fitting method, and more.
- Can batch-process multiple audio files and merge their SVGs into a single composite image.

## Features

- **Waveform to SVG**: Extracts the amplitude envelope and fits it using polynomial or Fourier methods.
- **Circular Mapping**: Maps the fitted curve onto a circle for unique radial visualizations.
- **Customizable Output**: Choose color, smoothness, fitting method, SVG size, and whether to output the perfect circle with the waveform.
- **Batch Processing**: Use shell scripts to process many files in parallel.
- **SVG Merging**: Merge and scale multiple SVGs into a single SVG/PNG for comparison or collage.

---

## Getting Started

1. **Clone the repository:**

   ```bash
   git clone https://github.com/JakubMifek/ripple.git
   cd ripple
   ```

2. **Set up a virtual environment (recommended):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Convert a WAV file to SVG

```bash
python main.py <input_wav> <output_svg> [options]
```

#### Options

- `--mode`: Mode of operation (default: Ripple, no other available atm)
- `--smoothness`: Waveform smoothing (0=raw, 1=flat, default: 0.3333)
- `--poly-degree`: Degree/terms for polynomial/Fourier fitting (default: 16)
- `--max-width`: Maximum SVG width (default: 4096)
- `--height`: SVG height (default: 512)
- `--fitting`: Curve fitting method: `fourier` or `poly` (default: fourier)
- `--color`: Color of the output curve (default: orange, accepts any SVG color)
- `--circle`: If set, adds a perfect circle as SVG to the outputted waveform; otherwise, output only the waveform curve

#### Usage Example

```bash
python main.py samples/HelloWorld.wav output_samples/HelloWorld.svg --color "#00aaff" --circle
```

This will generate a circular SVG visualization of `HelloWorld.wav` with a blue curve.

---

### 2. Batch Processing and Merging

You can process multiple WAV files in parallel and merge their SVGs using the provided `names.sh` script:

```bash
bash names.sh
```

The script will:

- Convert each WAV file in `samples/` to an SVG in `output_names/` with a specified color and style.
- Wait for all conversions to finish.
- Merge all SVGs in `output_names/` into a single composite SVG and PNG using `merge.py`.

---

### 3. Merge and Scale SVGs

You can merge and scale SVGs from a directory into a single SVG and PNG:

```bash
python merge.py <directory> [--width WIDTH] [--height HEIGHT] [--output OUTPUT_SVG] [--png OUTPUT_PNG]
```

#### Example

```bash
python merge.py output_names --width 2048 --height 2048 --output merged.svg --png merged.png
```

---

## Output

- **SVG**: The main output is an SVG file visualizing the waveform or its circular mapping.
- **PNG**: When merging, a PNG is also produced if `cairosvg` is installed.
- **tmp.svg**: A temporary SVG with the linear waveform is always written for reference.

---

## Tips & Notes

- The tool works best with 16-bit PCM WAV files.
- For stereo files, channels are automatically combined to mono.
- Colors can be any valid SVG color (e.g., `red`, `#ff0000`, `rgb(0,255,0)`).
- Batch processing is parallelized for speed.

---

## License

This project is licensed under the MIT License, a permissive open-source license. You are free to use, modify, and distribute the code, even for commercial purposes, as long as you include the original copyright notice. See [LICENSE](LICENSE) for details.
