import React, { useEffect, useState } from 'react';
import pt from 'prop-types';
import { Container, Grid } from '@mui/material';
import { useSelector, useDispatch } from 'react-redux';
import Plot from 'react-plotly.js';

import { IndicatorPlot } from '../plots';

const summaryData = [
  {
    name: 'KNN',
    metrics: {
      'Recall@20': 0.3,
      'Recall@50': 0.4,
      'NDCG@100': 0.1,
      'Coverage@20': 0.3,
      'Coverage@50': 0.8,
      'Coverage@100': 0.2,
    },
  },
  {
    name: 'SVD',
    metrics: {
      'Recall@20': 0.2,
      'Recall@50': 0.5,
      'NDCG@100': 0.23,
      'Coverage@20': 0.36,
      'Coverage@50': 0.78,
      'Coverage@100': 0.23,
    },
  },
  {
    name: 'VASP',
    metrics: {
      'Recall@20': 0.5,
      'Recall@50': 0.6,
      'NDCG@100': 0.6,
      'Coverage@20': 0.6,
      'Coverage@50': 0.8,
      'Coverage@100': 0.5,
    },
  },
];

var x = [];
for (var i = 0; i < 500; i++) {
  x[i] = Math.random();
}

function ModelsEvaluation() {
  return (
    <Container maxWidth="xl">
      <Grid container direction="column">
        <Grid item>
          <Plot
            data={summaryData.map((model) => ({
              x: Object.keys(model.metrics),
              y: Object.values(model.metrics),
              name: model.name,
              type: 'bar',
            }))}
            layout={{ width: 800, height: 350, title: 'Models Evaluation Summary' }}
          />
        </Grid>
        <Grid item xs={12}>
          <Plot
            data={Object.entries(summaryData[0].metrics).map(([metric, value], index) => ({
              value: value * 100,
              title: { text: metric },
              type: 'indicator',
              mode: 'gauge+number',
              gauge: { axis: { range: [0, 100] } },
              domain: { row: 0, column: index },
            }))}
            layout={{
              // width: 1400,
              // height: 300,
              autosize: true,
              // paper_bgcolor: '#fafafa',
              margin: { t: 25, b: 25, l: 25, r: 25 },
              grid: {
                rows: 1,
                columns: Object.keys(summaryData[0].metrics).length,
                pattern: 'independent',
              },
            }}
          />
        </Grid>
      </Grid>
    </Container>
  );
}

export default ModelsEvaluation;
