import pandas as pd
import holoviews as hv
from holoviews import opts
from bokeh.plotting import show
from bokeh.io import output_file

# Enable Bokeh backend for Holoviews
hv.extension('bokeh')

# Load the data
data = {
    'source': [
        'NS Program',
        'NS Program',
        'NS Program',
        'NS Program',
        'NS Program',
        'NS Program',
        'Predicate: terrain',
        'Predicate: terrain',
        'Predicate: far',
        'Predicate: far',
        'Predicate: far',
        'Predicate: in_way',
        'Predicate: in_way',
        'Predicate: front',
        'Predicate: slope',
        'Grounded SAM',
        'Grounded SAM',
        'Grounded SAM',
        'Monocular PointCloud',
        'Monocular PointCloud',
        'Program Exec overhead',
        'Label Matching',
        'Unify per-class\nmasks',
        'Pinhole Camera\nBack-project',
        'Object Pix2Pix\nDistance Eval',
        'Retrieve Terrain cache',
        'Traversability\nPix2Pix Eval',
        'Retrieve Distance cache',
        'Retrieve PointCloud cache',
        'TerrainSeg NN',
        'Grounding DINO',
        'SAM',
        'Metric DepthAnything'
    ],
    'target': [
        'Program Exec overhead',
        'Predicate: terrain',
        'Predicate: far',
        'Predicate: in_way',
        'Predicate: front',
        'Predicate: slope',
        'TerrainSeg NN',
        'Label Matching',
        'Grounded SAM',
        'Monocular PointCloud',
        'Object Pix2Pix\nDistance Eval',
        'Retrieve Terrain cache',
        'Traversability\nPix2Pix Eval',
        'Retrieve Distance cache',
        'Retrieve PointCloud cache',
        'Grounding DINO',
        'SAM',
        'Unify per-class\nmasks',
        'Metric DepthAnything',
        'Pinhole Camera\nBack-project',
        'Symbolic Exec',
        'Symbolic Exec',
        'Symbolic Exec',
        'Symbolic Exec',
        'Symbolic Exec',
        'Symbolic Exec',
        'Symbolic Exec',
        'Symbolic Exec',
        'Symbolic Exec',
        'Neural Exec',
        'Neural Exec',
        'Neural Exec',
        'Neural Exec'
    ],
    'value': [
        3,
        163,
        985,
        29,
        5,
        5,
        160,
        3,
        740,
        190,
        55,
        4,
        25,
        5,
        5,
        240,
        270,
        230,
        145,
        45,
        3,
        3,
        230,
        45,
        55,
        4,
        25,
        5,
        5,
        160,
        240,
        270,
        145
    ]
}

# for i in range(len(data['source'])):
#     data['source'][i] += '\n'
#     data['target'][i] += '\n'

edges = pd.DataFrame(data)

# Create Sankey diagram
sankey = hv.Sankey(edges, kdims=['source', 'target'], vdims='value')


def hook(plot, element):
    text_glyph = plot.handles['text_1_glyph']
    text_glyph.text_font_size = '11pt'  # Set to desired font size
    # text_glyph.text_font_style = 'bold'
    # text_glyph.text_align = 'center'
    # text_glyph.text_outline_color = 'white'
    text_glyph.text_color = 'black'

    plot.handles['text_1_glyph_renderer'].glyph = text_glyph

    text_data = plot.handles['text_1_source'].data
    for i, text in enumerate(text_data['text']):
        text_data['text'][i] = text.replace(' - ', ' (')
        text_data['text'][i] += ')'


# Customize the Sankey diagram
sankey.opts(
    opts.Sankey(
        width=1100,
        height=700,
        label_position='left',
        edge_color='source',
        node_color='index',
        cmap='tab20',
        # node_line_width=0,
        node_line_color='index',
        # label_text_font_size='9pt',
        hooks=[hook]
    )
)

# # Output to an HTML file
# output_file("/home/dynamo/AMRL_Research/repos/nspl/scripts/safety/realtime/sankey_diagram.html")

# # Show the plot
# show(hv.render(sankey))

hv.save(sankey, '/home/dynamo/AMRL_Research/repos/nspl/scripts/safety/realtime/sankey_diagram4.png', fmt='png')
