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
          number: { suffix: '%' },
          title: { text: title, font: { size: 16 } },
          delta: { reference: delta },
          type: 'indicator',
          mode: 'gauge+number+delta',
          gauge: { axis: { range: [min, max] } },
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
        margin: { b: 0, t: 50, l: 40, r: 40 },
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
  max: 100,
  delta: 0,
  height: 200,
};

export default IndicatorPlot;
