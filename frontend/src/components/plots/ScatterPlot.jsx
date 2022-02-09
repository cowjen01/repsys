import React from 'react';
import pt from 'prop-types';
import Plot from 'react-plotly.js';
import { useTheme } from '@mui/material/styles';

function ScatterPlot({ x, y, label, meta, innerRef, color, height, ...props }) {
  const theme = useTheme();
  const { text } = theme.palette;
  return (
    <Plot
      style={{
        width: '100%',
        height,
      }}
      useResizeHandler
      ref={innerRef}
      data={[
        {
          x,
          y,
          hovertext: label,
          hoverinfo: 'text',
          type: 'scatter',
          mode: 'markers',
          customdata: meta,
          unselected: {
            marker: {
              opacity: 1,
              color: '#e8eaf6',
            },
          },
          marker: {
            size: 4,
            color,
          },
        },
      ]}
      layout={{
        autosize: true,
        dragmode: 'lasso',
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: text.primary },
        uirevision: true,
        xaxis: { zeroline: false },
        yaxis: { zeroline: false },
        margin: { t: 20, b: 20, l: 20, r: 20 },
      }}
      {...props}
    />
  );
}

ScatterPlot.propTypes = {
  x: pt.arrayOf(pt.number).isRequired,
  y: pt.arrayOf(pt.number).isRequired,
  label: pt.arrayOf(pt.string),
  color: pt.oneOfType([pt.arrayOf(pt.oneOfType([pt.number, pt.string])), pt.string]),
  height: pt.oneOfType([pt.number, pt.string]),
  meta: pt.arrayOf(pt.any),
  // eslint-disable-next-line react/forbid-prop-types
  innerRef: pt.any,
};

ScatterPlot.defaultProps = {
  height: '100%',
  color: [],
  label: [],
  meta: [],
  innerRef: null,
};

export default ScatterPlot;
