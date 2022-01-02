import React from 'react';
import pt from 'prop-types';
import Plot from 'react-plotly.js';
import { useTheme } from '@mui/material/styles';

function BarPlot({ data, height, ...props }) {
  const theme = useTheme();
  const { text } = theme.palette;
  return (
    <Plot
      data={data.map((bar) => ({ ...bar, type: 'bar', orientation: 'h' }))}
      style={{
        width: '100%',
        height,
      }}
      useResizeHandler
      layout={{
        autosize: true,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: text.primary },
        margin: { t: 30, b: 40, l: 120, r: 40 },
      }}
      {...props}
    />
  );
}

BarPlot.propTypes = {
  data: pt.arrayOf(
    pt.shape({
      x: pt.arrayOf(pt.number),
      y: pt.arrayOf(pt.string),
      name: pt.string,
    })
  ).isRequired,
  height: pt.oneOfType([pt.number, pt.string]),
};

BarPlot.defaultProps = {
  height: '100%',
};

export default BarPlot;
