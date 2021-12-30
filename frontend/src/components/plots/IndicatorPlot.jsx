import React from 'react';
import pt from 'prop-types';
import Plot from 'react-plotly.js';

function IndicatorPlot({ title, value, min, max, width, height }) {
  return (
    <Plot
      data={[
        {
          value,
          title: { text: title },
          type: 'indicator',
          mode: 'gauge+number',
          gauge: { axis: { range: [min, max] } },
        },
      ]}
      layout={{ width, height, margin: { b: 10, t: 50, l: 40, r: 40 } }}
    />
  );
}

IndicatorPlot.propTypes = {
  title: pt.string.isRequired,
  value: pt.number.isRequired,
  min: pt.number,
  max: pt.number,
  width: pt.number,
  height: pt.number,
};

IndicatorPlot.defaultProps = {
  min: 0,
  max: 100,
  width: 200,
  height: 150,
};

export default IndicatorPlot;
