import React, { useState } from 'react';
import pt from 'prop-types';
import Box from '@mui/material/Box';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  Bar,
  BarChart,
} from 'recharts';
import Typography from '@mui/material/Typography';
import Grid from '@mui/material/Grid';

const data = [
  {
    name: 'Recall@20',
    uv: 0.419,
    pv: 0.448,
  },
  {
    name: 'Recall@50',
    uv: 0.387,
    pv: 0.414,
  },
  {
    name: 'NDCG@100',
    uv: 0.524,
    pv: 0.552,
  },
];

const data2 = [
  {
    uv: 0.419,
  },
  {
    uv: 0.387,
  },
  {
    uv: 0.524,
  },
  {
    uv: 0.287,
  },
  {
    uv: 0.124,
  },
];

const data3 = [
  { x: 100, y: 200, z: 200 },
  { x: 120, y: 100, z: 260 },
  { x: 170, y: 300, z: 400 },
  { x: 140, y: 250, z: 280 },
  { x: 150, y: 400, z: 500 },
  { x: 110, y: 280, z: 200 },
];

function ModelMetrics() {
  return (
    <Box
      sx={{
        p: 2,
      }}
    >
      <Grid container spacing={2} justifyContent="center" flexDirection="row">
        <Grid item>
          <Typography variant="subtitle1" textAlign="center" gutterBottom>
            Model Benchmarks
          </Typography>
          <BarChart width={400} height={250} data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="uv" name="Baseline" fill="#212121" radius={[5, 5, 0, 0]} />
            <Bar dataKey="pv" name="VASP" fill="#ff3d00" radius={[5, 5, 0, 0]} />
          </BarChart>
        </Grid>
        <Grid item>
          <Typography variant="subtitle1" textAlign="center" gutterBottom>
            Popularity Bias
          </Typography>
          <BarChart width={400} height={250} data={data2}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis />
            <YAxis />
            <Tooltip />
            <Bar dataKey="uv" fill="#212121" radius={[5, 5, 0, 0]} />
          </BarChart>
        </Grid>
        <Grid item>
          <Typography variant="subtitle1" textAlign="center" gutterBottom>
            Popularity Distribution
          </Typography>
          <ScatterChart
            width={400}
            height={250}
          >
            <CartesianGrid />
            <XAxis type="number" dataKey="x" name="stature" />
            <YAxis type="number" dataKey="y" name="weight" />
            <Tooltip cursor={{ strokeDasharray: '3 3' }} />
            <Scatter data={data3} fill="#212121" />
          </ScatterChart>
        </Grid>
      </Grid>
    </Box>
  );
}

export default ModelMetrics;
