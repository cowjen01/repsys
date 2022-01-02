import React from 'react';
import pt from 'prop-types';
import Plot from 'react-plotly.js';
import { useTheme } from '@mui/material/styles';

function HistogramPlot({ data, innerRef, height, ...props }) {
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
      ref={innerRef}
      useResizeHandler
      layout={{
        dragmode: 'select',
        selectdirection: 'h',
        autosize: true,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: text.primary },
        uirevision: true,
        margin: { t: 20, b: 20, l: 20, r: 20 },
      }}
      {...props}
    />
  );
}

HistogramPlot.propTypes = {
  data: pt.arrayOf(pt.number).isRequired,
  height: pt.oneOfType([pt.number, pt.string]),
  // eslint-disable-next-line react/forbid-prop-types
  innerRef: pt.any,
};

HistogramPlot.defaultProps = {
  height: '100%',
  innerRef: null,
};

export default HistogramPlot;
