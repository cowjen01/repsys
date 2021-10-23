import React, { useState, useEffect, useMemo } from 'react';
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
import Chart from 'react-google-charts';

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

const colors = ['#2196f3', '#03a9f4', '#4caf50', '#9c27b0', '#ff9800', '#cddc39', '#ffeb3b'];

function ModelMetrics() {
  const [scatterData, setScatterData] = useState([]);

  useEffect(() => {
    fetch('https://amp.pharm.mssm.edu/scavi/graph/GSE48968/tSNE/3')
      .then((response) => response.json())
      .then((d) => setScatterData(d));
  }, []);

  // const clusterData = useMemo(
  //   () =>
  //     scatterData.reduce((rv, x) => {
  //       (rv[x['KMeans-clustering']] = rv[x['KMeans-clustering']] || []).push(x);
  //       return rv;
  //     }, {}),
  //   [scatterData]
  // );

  const preparedData = useMemo(
    () => scatterData.map((p) => [p.x, p.y, `fill-color: ${colors[p['KMeans-clustering'] % 7]}`]),
    [scatterData]
  );

  // console.log(clusterData);

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
          <Chart
            width={400}
            height={400}
            chartType="ScatterChart"
            loader={<div>Loading Chart</div>}
            data={[['X', 'Y', { type: 'string', role: 'style' }], ...preparedData]}
            options={{
              title: 'Age vs. Weight comparison',
              dataOpacity: 1,
              enableInteractivity: true,
              pointSize: 3,
              legend: 'none',
              // explorer: {
              //   keepInBounds: true,
              //   maxZoomIn: 3,
              //   maxZoomOut: 1,
              //   zoomDelta: 1.1
              // }
            }}
            chartEvents={[
              {
                eventName: 'select',
                callback: ({ chartWrapper }) => {
                  const chart = chartWrapper.getChart();
                  const selection = chart.getSelection();
                  if (selection.length === 1) {
                    const [selectedItem] = selection;
                    const dataTable = chartWrapper.getDataTable();
                    const { row, column } = selectedItem;
                    alert(
                      'You selected : ' +
                        JSON.stringify({
                          row,
                          column,
                          value: dataTable.getValue(row, column),
                        }),
                      null,
                      2
                    );
                  }
                  console.log(selection);
                },
              },
            ]}
            rootProps={{ 'data-testid': '1' }}
          />
        </Grid>
      </Grid>
    </Box>
  );
}

export default ModelMetrics;
