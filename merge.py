
import os
import argparse
import xml.etree.ElementTree as ET
from typing import List
from svgpathtools import parse_path, Path

try:
    import cairosvg
except ImportError:
    cairosvg = None

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

def scale_path(path_elem: ET.Element, scale_x: float, scale_y: float) -> None:
    """Scale the d attribute of a path element by scale_x and scale_y using svgpathtools."""
    d_str = path_elem.attrib['d']
    path = parse_path(d_str)
    # Ensure path is always a Path object (not a single segment)
    if not isinstance(path, Path):
        path = Path(path)
    # Use svgpathtools' built-in scaling for the whole path
    scaled_path = path.scaled(scale_x, scale_y)
    # Ensure scaled_path is a Path object
    if not isinstance(scaled_path, Path):
        scaled_path = Path(scaled_path)
    # Use Path.d() if available, else fallback to join
    if hasattr(scaled_path, 'd'):
        path_elem.attrib['d'] = scaled_path.d()
    else:
        path_elem.attrib['d'] = ''.join(seg.d() for seg in scaled_path)

def main():
    parser = argparse.ArgumentParser(description="Merge and scale SVG paths from a directory into a single SVG and PNG.")
    parser.add_argument('directory', help='Directory containing SVG files')
    parser.add_argument('--width', type=int, default=4096, help='Output SVG width (default: 4096)')
    parser.add_argument('--height', type=int, default=4096, help='Output SVG height (default: 4096)')
    parser.add_argument('--output', type=str, default='output/merged.svg', help='Output SVG filename (default: merged.svg)')
    parser.add_argument('--png', type=str, default='output/merged.png', help='Output PNG filename (default: merged.png)')
    args = parser.parse_args()

    svg_files = [os.path.join(args.directory, f) for f in os.listdir(args.directory) if f.lower().endswith('.svg')]
    if not svg_files:
        print('No SVG files found in directory.')
        return

    all_paths = []
    for svg_file in svg_files:
        paths = parse_svg_paths(svg_file)
        if not paths:
            continue
        # Get original SVG size for scaling
        orig_w, orig_h = get_svg_size(svg_file)
        if orig_w == 0 or orig_h == 0:
            print(f'Warning: Could not determine size for {svg_file}, skipping.')
            continue
        scale_x = args.width / orig_w
        scale_y = args.height / orig_h
        for path in paths:
            scale_path(path, scale_x, scale_y)
            all_paths.append(path)

    # Create merged SVG
    svg_ns = 'http://www.w3.org/2000/svg'
    ET.register_namespace('', svg_ns)
    svg_root = ET.Element('{%s}svg' % svg_ns, width=str(args.width), height=str(args.height), version='1.1')
    for path in all_paths:
        svg_root.append(path)
    tree = ET.ElementTree(svg_root)
    tree.write(args.output, encoding='utf-8', xml_declaration=True)
    print(f'Merged SVG written to {args.output}')

    # Export to PNG if cairosvg is available
    if cairosvg:
        with open(args.output, 'rb') as f:
            svg_data = f.read()
        cairosvg.svg2png(bytestring=svg_data, write_to=args.png, output_width=args.width, output_height=args.height)
        print(f'Merged PNG written to {args.png}')
    else:
        print('cairosvg not installed, PNG export skipped.')

if __name__ == '__main__':
    main()
