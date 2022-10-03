"""
Module containing everything related to plotting: 
- hvplot extra functions
"""

import hvplot
from bokeh.io import export_svg
import holoviews as hv  
from cairosvg import svg2png

import amof.files.path 

def save_hvplot(plot, filename, format = 'svg+png'):
    """
    Save hvplot to filename

    Args:
        plot: hvplot object
        filename: string or pathlib object indicating where to save output
        format: str, can be 'svg+png' (high res png) or 'svg' or 'png' (low res)
    """
    if format[0:3] == 'svg':
        output_filename = str(amof.files.path.append_suffix(filename, 'svg'))
        bp = hv.render(plot)
        bp.output_backend = "svg"
        export_svg(bp, filename=output_filename)
        if format == 'svg+png':
            svg_code = open(output_filename, 'rt').read()
            png_filename = str(amof.files.path.append_suffix(filename, 'png'))
            svg2png(bytestring=svg_code,write_to=png_filename, scale = 3) # default is 100
    elif format == 'png':
        output_filename = str(amof.files.path.append_suffix(filename, 'png'))
        hvplot.save(plot, output_filename)
    else:
        raise ValueError('Format not supported')