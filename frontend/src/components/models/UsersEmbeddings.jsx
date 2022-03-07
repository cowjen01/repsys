import React, { useState, useMemo } from 'react';
import pt from 'prop-types';
import { Paper, Grid, Box, Stack } from '@mui/material';

import UsersDescription from '../dataset/UsersDescription';
import { useGetUserMetricsByModelQuery, useGetUsersEmbeddingsQuery } from '../../api';
import { CategoryFilter } from '../filters';
import EmbeddingsPlot from '../dataset/EmbeddingsPlot';
import { PlotLoader } from '../loaders';

function UsersEmbeddings({ attributes, metricsData }) {
  const models = Object.keys(metricsData.results);
  const metrics = metricsData.metrics.user;

  const [selectedUsers, setSelectedUsers] = useState([]);
  const [selectedMarkerSize, setSelectedMarkerSize] = useState(4);
  const [selectedModel, setSelectedModel] = useState(models[0]);
  const [selectedMetric, setSelectedMetric] = useState(metrics[0]);
  const [plotResetIndex, setPlotResetIndex] = useState(0);
  const [selectedColorScale, setSelectedColorScale] = useState('Picnic');

  const embeddings = useGetUsersEmbeddingsQuery('validation');
  const userMetrics = useGetUserMetricsByModelQuery(selectedModel);

  const handleModelChange = (newValue) => {
    setSelectedModel(newValue);
    // setSelectedMetric(metrics[0]);
    setPlotResetIndex(plotResetIndex + 1);
    setSelectedUsers([]);
  };

  const handleMetricChange = (newValue) => {
    setSelectedMetric(newValue);
    setPlotResetIndex(plotResetIndex + 1);
    setSelectedUsers([]);
  };

  const handlePlotUnselect = () => {
    setSelectedUsers([]);
  };

  const handlePlotSelect = (ids) => {
    setSelectedUsers(ids);
  };

  const embeddingsColor = useMemo(() => {
    if (userMetrics.data) {
      return userMetrics.data.map((d) => d[selectedMetric]);
    }
    return [];
  }, [userMetrics.data, selectedMetric]);

  return (
    <Grid container spacing={2}>
      <Grid item xs={12}>
        <Stack direction="row" spacing={1}>
          <CategoryFilter
            label="Model"
            value={selectedModel}
            onChange={handleModelChange}
            options={models}
          />
          <CategoryFilter
            label="Metric"
            disabled={userMetrics.isFetching}
            value={selectedMetric}
            onChange={handleMetricChange}
            options={metrics}
          />
          <CategoryFilter
            label="Color scale"
            value={selectedColorScale}
            onChange={setSelectedColorScale}
            options={['Picnic', 'Bluered', 'Jet', 'RdBu']}
          />
          <CategoryFilter
            label="Marker size"
            value={selectedMarkerSize}
            onChange={setSelectedMarkerSize}
            options={[2, 3, 4, 5, 6]}
          />
        </Stack>
      </Grid>
      <Grid item xs={12}>
        <Grid container spacing={2} sx={{ height: 500 }}>
          <Grid item xs={8}>
            <Box position="relative">
              {userMetrics.isFetching && <PlotLoader />}
              <Paper sx={{ p: 2 }}>
                <EmbeddingsPlot
                  onUnselect={handlePlotUnselect}
                  embeddings={embeddings.data}
                  onSelect={handlePlotSelect}
                  color={embeddingsColor}
                  resetIndex={plotResetIndex}
                  colorScale={selectedColorScale}
                  markerSize={selectedMarkerSize}
                  showScale
                />
              </Paper>
            </Box>
          </Grid>
          <Grid item xs={4} sx={{ height: '100%' }}>
            {selectedUsers.length > 0 && (
              <Paper sx={{ p: 2, height: '100%', overflow: 'auto' }}>
                <UsersDescription
                  attributes={attributes}
                  split="validation"
                  users={selectedUsers}
                />
              </Paper>
            )}
          </Grid>
        </Grid>
      </Grid>
    </Grid>
  );
}

UsersEmbeddings.propTypes = {
  metricsData: pt.any.isRequired,
  attributes: pt.any.isRequired,
};

export default UsersEmbeddings;
