
import os
import argparse
import sys
import xml.etree.ElementTree as ET
from typing import List
from svgpathtools import parse_path, Path
from config import REFERENCE_STROKE_WIDTH
from utils import reference_first

def parse_svg_paths(svg_file: str) -> List[ET.Element]:
    """Parse all <path> elements from an SVG file."""
    tree = ET.parse(svg_file)
    root = tree.getroot()
    # SVG namespace handling
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    paths = root.findall('.//svg:path', ns)
    if not paths:
        # fallback: try without namespace
        paths = root.findall('.//path')
    return paths

def get_svg_size(svg_file: str) -> tuple[int, int]:
    """Extract width and height from SVG root attributes."""
    tree = ET.parse(svg_file)
    root = tree.getroot()
    width = root.attrib.get('width')
    height = root.attrib.get('height')
    def parse_dim(val):
        if val is None:
            return 0
        if val.endswith('px'):
            return int(float(val[:-2]))
        return int(float(val))
    return parse_dim(width), parse_dim(height)

def scale_path(path_elem: ET.Element, scale_x: float, scale_y: float, thickness: int) -> None:
    """Scale the d attribute of a path element by scale_x and scale_y using svgpathtools."""
    path_elem.attrib['stroke-width'] = str(REFERENCE_STROKE_WIDTH if path_elem.get('data-role') == 'reference-circle' else thickness)
    d_str = path_elem.attrib['d']
    path = parse_path(d_str)
    # Ensure path is always a Path object (not a single segment)
    if not isinstance(path, Path):
        path = Path(path)
    # Use svgpathtools' built-in scaling for the whole path
    scaled_path = path.scaled(scale_x, scale_y, origin=0j)
    # Ensure scaled_path is a Path object
    if not isinstance(scaled_path, Path):
        scaled_path = Path(scaled_path)
    # Use Path.d() if available, else fallback to join
    if hasattr(scaled_path, 'd'):
        path_elem.attrib['d'] = scaled_path.d()
    else:
        path_elem.attrib['d'] = ''.join(seg.d() for seg in scaled_path)

def export_png(svg_path, png_path, width, height, required=False):
    """Missing Python/native Cairo is optional only for the implicit default export."""
    try:
        import cairosvg
    except (ImportError, OSError) as error:
        if required:
            raise RuntimeError(f"PNG requested but CairoSVG/native Cairo is unavailable: {error}") from error
        print(f"Warning: PNG export skipped; CairoSVG/native Cairo unavailable: {error}", file=sys.stderr)
        return False
    try:
        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path),
                        output_width=width, output_height=height)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"PNG rendering failed: {error}") from error
    return True


def _main(argv=None):
    parser = argparse.ArgumentParser(description="Merge and scale SVG paths from a directory into a single SVG and PNG.")
    parser.add_argument('directory', help='Directory containing SVG files')
    parser.add_argument('--width', type=int, default=4096, help='Output SVG width (default: 4096)')
    parser.add_argument('--height', type=int, default=4096, help='Output SVG height (default: 4096)')
    parser.add_argument('--output', type=str, required=True, help='Explicit output SVG filename, outside the input directory')
    parser.add_argument('--thickness', type=int, default=2,
                        help=f'Artwork stroke thickness (default: 2); reference circles stay {REFERENCE_STROKE_WIDTH} px')
    parser.add_argument('--png', type=str, help='Explicit PNG filename (failure is an error); implicit default: OUTPUT_STEM.png')
    parser.add_argument('--no-png', action='store_true', help='Only write SVG')
    args = parser.parse_args(argv)
    if args.width <= 0 or args.height <= 0 or args.thickness <= 0:
        parser.error("Dimensions and thickness must be positive.")
    if args.no_png and args.png:
        parser.error("--png and --no-png cannot be combined.")
    if os.path.dirname(os.path.abspath(args.output)) == os.path.abspath(args.directory):
        parser.error("Keep merged output outside the input directory.")
    png = args.png or os.path.splitext(args.output)[0] + ".png"
    if not args.no_png and os.path.abspath(png) == os.path.abspath(args.output):
        parser.error("SVG and PNG outputs must be distinct.")

    svg_files = sorted(os.path.join(args.directory, f) for f in os.listdir(args.directory) if f.lower().endswith('.svg'))
    if not svg_files:
        parser.error('No SVG files found in directory.')
    if not args.no_png and os.path.abspath(png) in {os.path.abspath(p) for p in svg_files}:
        parser.error("PNG output must not overwrite an input SVG.")

    all_paths = []
    for svg_file in svg_files:
        paths = parse_svg_paths(svg_file)
        if not paths:
            parser.error(f"No SVG paths found in {svg_file}.")
        # Get original SVG size for scaling
        orig_w, orig_h = get_svg_size(svg_file)
        if orig_w <= 0 or orig_h <= 0:
            parser.error(f'Could not determine positive size for {svg_file}.')
        scale_x = args.width / orig_w
        scale_y = args.height / orig_h
        for path in paths:
            scale_path(path, scale_x, scale_y, args.thickness)
            all_paths.append(path)

    # Create merged SVG
    svg_ns = 'http://www.w3.org/2000/svg'
    ET.register_namespace('', svg_ns)
    svg_root = ET.Element('{%s}svg' % svg_ns, width=str(args.width), height=str(args.height), version='1.1')
    svg_root.extend(reference_first(all_paths))
    tree = ET.ElementTree(svg_root)
    tree.write(args.output, encoding='utf-8', xml_declaration=True)
    print(f'Merged SVG written to {args.output}')

    # Export to PNG if cairosvg is available
    if not args.no_png:
        try:
            if export_png(args.output, png, args.width, args.height, required=args.png is not None):
                print(f'Merged PNG written to {png}')
        except RuntimeError as error:
            parser.exit(2, f"Merge error: {error}\n")


def main(argv=None):
    try:
        _main(argv)
    except (OSError, ValueError, ET.ParseError) as error:
        raise SystemExit(f"Merge error: {error}") from error


if __name__ == '__main__':
    main()
