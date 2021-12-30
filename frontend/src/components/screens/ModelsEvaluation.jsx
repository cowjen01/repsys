import React, { useEffect, useState } from 'react';
import pt from 'prop-types';
import { Container, Grid, Paper, Typography } from '@mui/material';
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
      <Grid container direction="column" spacing={4}>
        <Grid item xs={12}>
          <Typography component="div" gutterBottom variant="h6">
            Models summary
          </Typography>
          <Plot
            data={summaryData.map((model) => ({
              x: Object.keys(model.metrics),
              y: Object.values(model.metrics),
              name: model.name,
              type: 'bar',
            }))}
            layout={{ width: 800, height: 350 }}
          />
        </Grid>
        {summaryData.map((model) => (
          <Grid item xs={12} key={model.name}>
            <Typography component="div" gutterBottom variant="h6">
              Model {model.name}
            </Typography>
            <Paper sx={{ p: 2 }}>
              {Object.entries(model.metrics).map(([metric, value]) => (
                <IndicatorPlot
                  width={200}
                  height={150}
                  key={metric}
                  title={metric}
                  value={value * 100}
                />
              ))}
            </Paper>
          </Grid>
        ))}
      </Grid>
    </Container>
  );
}

export default ModelsEvaluation;
