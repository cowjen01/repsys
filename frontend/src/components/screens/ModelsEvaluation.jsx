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
    metricsPrev: {
      'Recall@20': 0.2,
      'Recall@50': 0.5,
      'NDCG@100': 0.3,
      'Coverage@20': 0.1,
      'Coverage@50': 0.7,
      'Coverage@100': 0.15,
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
          <Grid container>
            <Grid item xs={6}>
              <Paper>
                <Plot
                  data={summaryData.map((model) => ({
                    x: Object.keys(model.metrics),
                    y: Object.values(model.metrics),
                    name: model.name,
                    type: 'bar',
                  }))}
                  layout={{
                    width: 700,
                    height: 300,
                    font: { color: '#fff' },
                    margin: { t: 40, l: 60 },
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                  }}
                />
              </Paper>
            </Grid>
          </Grid>
        </Grid>
        {summaryData.map((model) => (
          <Grid item xs={12} key={model.name}>
            <Typography component="div" gutterBottom variant="h6">
              Model {model.name}
            </Typography>
            <Paper>
              <Grid container>
                {Object.entries(model.metrics).map(([metric, value]) => (
                  <Grid item xs={2} key={metric}>
                    <IndicatorPlot
                      title={metric}
                      value={value * 100}
                      delta={
                        model.metricsPrev && model.metricsPrev[metric]
                          ? model.metricsPrev[metric] * 100
                          : 0
                      }
                    />
                  </Grid>
                ))}
              </Grid>
            </Paper>
          </Grid>
        ))}
      </Grid>
    </Container>
  );
}

export default ModelsEvaluation;
