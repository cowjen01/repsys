import React from 'react';
import pt from 'prop-types';
import Plot from 'react-plotly.js';
import { useTheme } from '@mui/material/styles';

function IndicatorPlot({ title, value, min, max, height, delta }) {
  const theme = useTheme();
  const { text } = theme.palette;
  return (
    <Plot
      data={[
        {
          value,
          title: { text: title, font: { size: 13 } },
          delta: delta ? { reference: delta } : null,
          type: 'indicator',
          mode: 'gauge+number+delta',
          gauge: { axis: { range: [min, max] } },
          hovermode: false,
        },
      ]}
      style={{
        width: '100%',
        height,
      }}
      useResizeHandler
      layout={{
        autosize: true,
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: { color: text.primary },
        margin: { b: 0, t: 30, l: 30, r: 30 },
      }}
      config={{
        displayModeBar: false,
      }}
    />
  );
}

IndicatorPlot.propTypes = {
  title: pt.string.isRequired,
  value: pt.number.isRequired,
  min: pt.number,
  max: pt.number,
  height: pt.number,
  delta: pt.number,
};

IndicatorPlot.defaultProps = {
  min: 0,
  max: 1,
  delta: null,
  height: 200,
};

export default IndicatorPlot;
