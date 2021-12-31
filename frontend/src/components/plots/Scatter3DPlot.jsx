import React from 'react';
import pt from 'prop-types';
import Plot from 'react-plotly.js';
import { useTheme } from '@mui/material/styles';

function Scatter3DPlot({ x, y, z, color, height, ...props }) {
  const theme = useTheme();
  const { text } = theme.palette;
  return (
    <Plot
      style={{
        width: '100%',
        height,
      }}
      useResizeHandler
      data={[
        {
          x,
          y,
          z,
          type: 'scatter3d',
          mode: 'markers',
          marker: {
            size: 3,
            opacity: 0.8,
            color,
          },
        },
      ]}
      layout={{
        autosize: true,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: text.primary },
        margin: { t: 30, b: 30, l: 30, r: 30 },
      }}
      {...props}
    />
  );
}

Scatter3DPlot.propTypes = {
  x: pt.arrayOf(pt.number).isRequired,
  y: pt.arrayOf(pt.number).isRequired,
  z: pt.arrayOf(pt.number).isRequired,
  color: pt.arrayOf(pt.number),
  height: pt.oneOfType([pt.number, pt.string]),
};

Scatter3DPlot.defaultProps = {
  height: '100%',
  color: [],
};

export default Scatter3DPlot;
