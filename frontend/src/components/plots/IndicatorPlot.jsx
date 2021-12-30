import React, { useEffect, useState } from 'react';
import pt from 'prop-types';
import { Container, Grid } from '@mui/material';
import { useSelector, useDispatch } from 'react-redux';
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
      layout={{ width, height, paper_bgcolor: '#fafafa' }}
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
  width: 350,
  height: 300,
};

export default IndicatorPlot;
