import React from 'react';
import pt from 'prop-types';
import Plot from 'react-plotly.js';
import { useTheme } from '@mui/material/styles';

function BarPlot({ data, height, width, orientation, layoutProps, ...props }) {
  const theme = useTheme();
  const { text } = theme.palette;
  return (
    <Plot
      data={data.map((bar) => ({ ...bar, type: 'bar', orientation }))}
      style={{
        width,
        height,
      }}
      useResizeHandler
      layout={{
        autosize: true,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: text.primary },
        margin: { t: 20, b: 20, l: 20, r: 20 },
        ...layoutProps,
      }}
      {...props}
    />
  );
}

BarPlot.propTypes = {
  orientation: pt.string,
  data: pt.arrayOf(
    pt.shape({
      x: pt.arrayOf(pt.oneOfType([pt.number, pt.string])),
      y: pt.arrayOf(pt.oneOfType([pt.number, pt.string])),
      name: pt.string,
    })
  ).isRequired,
  height: pt.oneOfType([pt.number, pt.string]),
  width: pt.oneOfType([pt.number, pt.string]),
  // eslint-disable-next-line react/forbid-prop-types
  layoutProps: pt.any,
};

BarPlot.defaultProps = {
  orientation: 'v',
  height: '100%',
  width: '100%',
  layoutProps: {},
};

export default BarPlot;
