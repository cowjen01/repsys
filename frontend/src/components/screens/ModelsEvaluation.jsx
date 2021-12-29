import React, { useEffect, useState } from 'react';
import pt from 'prop-types';
import { Container, Grid } from '@mui/material';
import { useSelector, useDispatch } from 'react-redux';
import Plot from 'react-plotly.js';

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
            layout={{ width: 500, height: 350, title: 'Models Evaluation Summary' }}
          />
        </Grid>
        <Grid item>
          <Grid container>
            {Object.entries(summaryData[0].metrics).map((metric, value) => (
              <Grid item key={metric}>
                <Plot
                  data={[
                    {
                      domain: { x: [0, 100], y: [0, 100] },
                      value: value * 100,
                      title: { text: metric },
                      type: 'indicator',
                      mode: 'gauge+number',
                      delta: { reference: 400 },
                      gauge: { axis: { range: [0, 100] } },
                    },
                  ]}
                  // layout={{ width: 500, height: 350, title: 'Models Evaluation Summary' }}
                />
              </Grid>
            ))}
          </Grid>
        </Grid>
      </Grid>
    </Container>
  );
}

export default ModelsEvaluation;
