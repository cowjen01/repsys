import React, { useState, useMemo, useRef } from 'react';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Stack,
  List,
  Box,
  Tabs,
  Tab,
  ListItem,
  Alert,
  ListSubheader,
  ListItemText,
} from '@mui/material';

import { IndicatorPlot, ScatterPlot, BarPlot } from '../plots';
import { ItemListView } from '../items';
import TabPanel from '../TabPanel';
import MetricsHistogramPlot from './MetricsHistogramPlot';
import { plotColors } from '../../const';

const scatterData = [
  { x: 0.379, y: 0.289, z: 0.529, c: 7 },
  { x: 0.43, y: 0.164, z: 0.429, c: 10 },
  { x: 0.703, y: 0.211, z: 0.062, c: 5 },
  { x: 0.738, y: 0.91, z: 0.28, c: 10 },
  { x: 0.086, y: 0.651, z: 0.991, c: 1 },
  { x: 0.466, y: 0.008, z: 0.398, c: 7 },
  { x: 0.371, y: 0.373, z: 0.55, c: 1 },
  { x: 0.661, y: 0.644, z: 0.443, c: 7 },
  { x: 0.329, y: 0.508, z: 0.897, c: 9 },
  { x: 0.979, y: 0.72, z: 0.858, c: 9 },
  { x: 0.781, y: 0.529, z: 0.049, c: 10 },
  { x: 0.955, y: 0.361, z: 0.667, c: 1 },
  { x: 0.095, y: 0.111, z: 0.249, c: 4 },
  { x: 0.287, y: 0.101, z: 0.801, c: 3 },
  { x: 0.048, y: 0.508, z: 0.531, c: 2 },
  { x: 0.14, y: 0.216, z: 0.693, c: 9 },
  { x: 0.587, y: 0.692, z: 0.075, c: 6 },
  { x: 0.958, y: 0.609, z: 0.211, c: 1 },
  { x: 0.017, y: 0.382, z: 0.332, c: 9 },
  { x: 0.945, y: 0.634, z: 0.946, c: 10 },
  { x: 0.177, y: 0.756, z: 0.155, c: 10 },
  { x: 0.773, y: 0.053, z: 0.799, c: 3 },
  { x: 0.144, y: 0.276, z: 0.08, c: 3 },
  { x: 0.166, y: 0.125, z: 0.748, c: 2 },
  { x: 0.068, y: 0.246, z: 0.809, c: 8 },
  { x: 0.094, y: 0.807, z: 0.867, c: 9 },
  { x: 0.795, y: 0.818, z: 0.378, c: 6 },
  { x: 0.973, y: 0.21, z: 0.149, c: 6 },
  { x: 0.245, y: 0.55, z: 0.981, c: 4 },
  { x: 0.173, y: 0.867, z: 0.65, c: 9 },
  { x: 0.372, y: 0.514, z: 0.765, c: 5 },
  { x: 0.795, y: 0.934, z: 0.391, c: 2 },
  { x: 0.498, y: 0.917, z: 0.138, c: 8 },
  { x: 0.027, y: 0.232, z: 0.61, c: 1 },
  { x: 0.673, y: 0.244, z: 0.184, c: 8 },
  { x: 0.427, y: 0.217, z: 0.321, c: 4 },
  { x: 0.639, y: 0.847, z: 0.195, c: 1 },
  { x: 0.598, y: 0.337, z: 0.19, c: 8 },
  { x: 0.872, y: 0.415, z: 0.553, c: 7 },
  { x: 0.104, y: 0.693, z: 0.991, c: 2 },
  { x: 0.328, y: 0.772, z: 0.704, c: 8 },
  { x: 0.676, y: 0.2, z: 0.21, c: 4 },
  { x: 0.126, y: 0.915, z: 0.232, c: 2 },
  { x: 0.9, y: 0.564, z: 0.281, c: 3 },
  { x: 0.178, y: 0.33, z: 0.096, c: 1 },
  { x: 0.541, y: 0.324, z: 0.154, c: 3 },
  { x: 0.31, y: 0.029, z: 0.873, c: 6 },
  { x: 0.045, y: 0.79, z: 0.608, c: 4 },
  { x: 0.266, y: 0.014, z: 0.224, c: 5 },
  { x: 0.959, y: 0.077, z: 0.708, c: 4 },
  { x: 0.479, y: 0.644, z: 0.505, c: 1 },
  { x: 0.334, y: 0.322, z: 0.859, c: 1 },
  { x: 0.104, y: 0.931, z: 0.031, c: 10 },
  { x: 0.136, y: 0.156, z: 0.453, c: 5 },
  { x: 0.59, y: 0.478, z: 0.686, c: 10 },
  { x: 0.225, y: 0.954, z: 0.021, c: 7 },
  { x: 0.328, y: 0.689, z: 0.466, c: 9 },
  { x: 0.546, y: 0.826, z: 0.788, c: 1 },
  { x: 0.508, y: 0.594, z: 0.652, c: 8 },
  { x: 0.505, y: 0.487, z: 0.759, c: 9 },
  { x: 0.67, y: 0.286, z: 0.19, c: 9 },
  { x: 0.52, y: 0.05, z: 0.781, c: 7 },
  { x: 0.232, y: 0.756, z: 0.669, c: 1 },
  { x: 0.347, y: 0.772, z: 0.353, c: 3 },
  { x: 0.157, y: 0.544, z: 0.512, c: 6 },
  { x: 0.042, y: 0.081, z: 0.385, c: 2 },
  { x: 0.139, y: 0.635, z: 0.484, c: 4 },
  { x: 0.816, y: 0.231, z: 0.352, c: 3 },
  { x: 0.427, y: 0.073, z: 0.227, c: 4 },
  { x: 0.593, y: 0.527, z: 0.304, c: 4 },
  { x: 0.757, y: 0.839, z: 0.219, c: 7 },
  { x: 0.451, y: 0.913, z: 0.362, c: 4 },
  { x: 0.897, y: 0.26, z: 0.865, c: 10 },
  { x: 0.708, y: 0.066, z: 0.11, c: 5 },
  { x: 0.146, y: 0.01, z: 0.465, c: 2 },
  { x: 0.459, y: 0.668, z: 0.136, c: 6 },
  { x: 0.776, y: 0.807, z: 0.841, c: 4 },
  { x: 0.192, y: 0.417, z: 0.405, c: 1 },
  { x: 0.449, y: 0.329, z: 0.134, c: 6 },
  { x: 0.046, y: 0.869, z: 0.692, c: 6 },
  { x: 0.999, y: 0.74, z: 0.319, c: 1 },
  { x: 0.064, y: 0.807, z: 0.813, c: 2 },
  { x: 0.337, y: 0.349, z: 0.614, c: 8 },
  { x: 0.962, y: 0.005, z: 0.028, c: 6 },
  { x: 0.747, y: 0.111, z: 0.674, c: 2 },
  { x: 0.456, y: 0.945, z: 0.24, c: 10 },
  { x: 0.442, y: 0.324, z: 0.132, c: 4 },
  { x: 0.041, y: 0.858, z: 0.062, c: 1 },
  { x: 0.196, y: 0.817, z: 0.696, c: 1 },
  { x: 0.383, y: 0.351, z: 0.607, c: 4 },
  { x: 0.906, y: 0.904, z: 0.268, c: 1 },
  { x: 0.062, y: 0.036, z: 0.244, c: 1 },
  { x: 0.813, y: 0.57, z: 0.888, c: 6 },
  { x: 1.0, y: 0.624, z: 0.429, c: 10 },
  { x: 0.334, y: 0.707, z: 0.825, c: 5 },
  { x: 0.282, y: 0.591, z: 0.207, c: 10 },
  { x: 0.072, y: 0.643, z: 0.248, c: 2 },
  { x: 0.372, y: 0.031, z: 0.498, c: 1 },
  { x: 0.963, y: 0.546, z: 0.5, c: 5 },
  { x: 0.705, y: 0.285, z: 0.659, c: 7 },
];

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
      'Novelty@10': 0.65,
      'Recall@100': 0.32,
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
      'Novelty@10': 0.32,
      'Recall@100': 0.42,
    },
  },
  {
    name: 'VASP',
    metrics: {
      'Recall@20': 0.2,
      'Recall@50': 0.5,
      'NDCG@100': 0.23,
      'Coverage@20': 0.36,
      'Coverage@50': 0.78,
      'Coverage@100': 0.23,
      'Novelty@10': 0.43,
      'Recall@100': 0.12,
    },
  },
];

function ModelsEvaluation() {
  const [selectedUsers, setSelectedUsers] = useState([]);

  const [histTab, setHistTab] = useState(0);
  const [modelTab, setModelTab] = useState(0);

  const handleHistTabChange = (event, newValue) => {
    setHistTab(newValue);
  };

  const handleModelTabChange = (e, modelIndex) => {
    setModelTab(modelIndex);
  };

  const scatterColors = useMemo(() => {
    if (selectedUsers.length === 0) {
      return [];
    }
    const colors = [];
    for (let i = 0; i < scatterData.length; i += 1) {
      colors.push(plotColors.unselectedMarker);
    }
    selectedUsers.forEach((p) => {
      colors[p] = plotColors.selectedMarker;
    });
    return colors;
  }, [selectedUsers]);

  const scatterPoints = useMemo(
    () => ({
      x: scatterData.map(({ x }) => x),
      y: scatterData.map(({ y }) => y),
      z: scatterData.map(({ z }) => z),
    }),
    []
  );

  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Box pl={1}>
          <Typography component="div" variant="h6">
            Models Performance
          </Typography>
          <Typography variant="subtitle1" gutterBottom>
            A performance in the individual metrics with comparasion to the previous evaluation
          </Typography>
        </Box>
        <Grid container spacing={2}>
          <Grid item xs={7}>
            <Paper sx={{ height: '100%' }}>
              <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                <Tabs value={modelTab} onChange={handleModelTabChange} variant="fullWidth">
                  {summaryData.map((m) => (
                    <Tab label={m.name} key={m.name} />
                  ))}
                </Tabs>
              </Box>
              <Box sx={{ p: 2 }}>
                <Grid container>
                  {Object.entries(summaryData[modelTab].metrics).map(([metric, value]) => (
                    <Grid item xs={3} key={metric}>
                      <IndicatorPlot
                        title={metric}
                        height={150}
                        value={value}
                        delta={
                          summaryData[0].metricsPrev && summaryData[0].metricsPrev[metric]
                            ? summaryData[0].metricsPrev[metric]
                            : 0
                        }
                      />
                    </Grid>
                  ))}
                </Grid>
              </Box>
            </Paper>
          </Grid>
          {summaryData.length > 1 && (
            <Grid item xs={5}>
              <Paper>
                <BarPlot
                  orientation="h"
                  data={summaryData.map((model) => ({
                    y: Object.keys(model.metrics),
                    x: Object.values(model.metrics),
                    name: model.name,
                  }))}
                  layoutProps={{
                    margin: { t: 30, b: 40, l: 120, r: 40 },
                  }}
                  height={400}
                />
              </Paper>
            </Grid>
          )}
        </Grid>
      </Grid>
      <Grid item xs={12}>
        <Box pl={1}>
          <Typography component="div" variant="h6">
            Metrics Distribution
          </Typography>
          <Typography variant="subtitle1" gutterBottom>
            A distribution of the metrics for each validation user
          </Typography>
        </Box>
        <Grid container spacing={2}>
          <Grid item xs={8}>
            <MetricsHistogramPlot onSelect={(users) => setSelectedUsers(users.points)} />
          </Grid>
          <Grid item xs={4}>
            {selectedUsers.length > 0 && (
              <Paper sx={{ height: '100%' }}>
                <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                  <Tabs value={histTab} onChange={handleHistTabChange} variant="fullWidth">
                    <Tab label="User embeddings" />
                    <Tab label="User details" />
                  </Tabs>
                </Box>

                <TabPanel value={histTab} index={0}>
                  <ScatterPlot
                    height={380}
                    x={scatterPoints.x}
                    y={scatterPoints.y}
                    z={scatterPoints.z}
                    color={scatterColors}
                  />
                </TabPanel>
                <TabPanel value={histTab} index={1}>
                  <List
                    dense
                    subheader={
                      <ListSubheader component="div">The Most Interacted Items</ListSubheader>
                    }
                  >
                    <ItemListView
                      title="Four Weddings and a Funeral (1994)"
                      subtitle="Comedy, Drama, Romance"
                      image="https://m.media-amazon.com/images/M/MV5BMTMyNzg2NzgxNV5BMl5BanBnXkFtZTcwMTcxNzczNA@@..jpg"
                    />
                    <ItemListView
                      title="Cutthroat Island (1995)"
                      subtitle="Action, Adventure, Comedy"
                    />
                    <ItemListView
                      title="Four Weddings and a Funeral (1994)"
                      subtitle="Comedy, Drama, Romance"
                      image="https://m.media-amazon.com/images/M/MV5BMTMyNzg2NzgxNV5BMl5BanBnXkFtZTcwMTcxNzczNA@@..jpg"
                    />
                    <ItemListView
                      title="Cutthroat Island (1995)"
                      subtitle="Action, Adventure, Comedy"
                    />
                    <ItemListView
                      title="Cutthroat Island (1995)"
                      subtitle="Action, Adventure, Comedy"
                    />
                  </List>
                </TabPanel>
              </Paper>
            )}
          </Grid>
        </Grid>
      </Grid>
    </Grid>
  );
}

export default ModelsEvaluation;
