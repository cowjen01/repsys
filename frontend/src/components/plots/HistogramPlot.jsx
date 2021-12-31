import React from 'react';
import pt from 'prop-types';
import Plot from 'react-plotly.js';
import { useTheme } from '@mui/material/styles';

function HistogramPlot({ data, height, ...props }) {
  const theme = useTheme();
  const { text } = theme.palette;
  return (
    <Plot
      data={[
        {
          x: data,
          type: 'histogram',
          opacity: 1,
          showlegend: false,
        },
      ]}
      style={{
        width: '100%',
        height,
      }}
      useResizeHandler
      layout={{
        dragmode: 'select',
        selectdirection: 'h',
        autosize: true,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: text.primary },
        margin: { t: 40, b: 40, l: 40, r: 40 },
      }}
      {...props}
    />
  );
}

HistogramPlot.propTypes = {
  data: pt.arrayOf(pt.number).isRequired,
  height: pt.oneOfType([pt.number, pt.string]),
};

HistogramPlot.defaultProps = {
  height: '100%',
};

export default HistogramPlot;
