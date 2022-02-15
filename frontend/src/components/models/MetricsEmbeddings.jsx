import React, { useState, useMemo } from 'react';
import pt from 'prop-types';
import { Paper, Grid, Box, Stack } from '@mui/material';

import UsersDescription from '../dataset/UsersDescription';
import { useGetMetricsByModelQuery, useGetUsersEmbeddingsQuery } from '../../api';
import { CategoryFilter } from '../filters';
import EmbeddingsPlot from '../dataset/EmbeddingsPlot';
import { PlotLoader } from '../loaders';

function MetricsEmbeddings({ metricsData }) {
  const models = Object.keys(metricsData.results);
  const metrics = metricsData.metrics.distributed.users;

  const [selectedUsers, setSelectedUsers] = useState([]);
  const [selectedModel, setSelectedModel] = useState(models[0]);
  const [selectedMetric, setSelectedMetric] = useState(metrics[0]);
  const [plotResetIndex, setPlotResetIndex] = useState(0);

  const embeddings = useGetUsersEmbeddingsQuery('validation');
  const modelMetrics = useGetMetricsByModelQuery(selectedModel);

  const handleModelChange = (newValue) => {
    setSelectedModel(newValue);
    setSelectedMetric(metrics[0]);
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
    if (modelMetrics.data) {
      return modelMetrics.data.map((d) => d[selectedMetric]);
    }
    return [];
  }, [modelMetrics.data, selectedMetric]);

  return (
    <Grid container spacing={2}>
      <Grid item xs={12}>
        <Stack direction="row" spacing={2}>
          <CategoryFilter
            label="Model"
            value={selectedModel}
            onChange={handleModelChange}
            options={models.map((model) => ({
              value: model,
              label: model,
            }))}
          />
          <CategoryFilter
            label="Metric"
            disabled={modelMetrics.isFetching}
            value={selectedMetric}
            onChange={handleMetricChange}
            options={metrics.map((metric) => ({
              value: metric,
              label: metric,
            }))}
          />
        </Stack>
      </Grid>
      <Grid item xs={12}>
        <Grid container spacing={2} sx={{ height: 500 }}>
          <Grid item xs={8}>
            <Box position="relative">
              {modelMetrics.isFetching && <PlotLoader />}
              <Paper sx={{ p: 2 }}>
                <EmbeddingsPlot
                  onUnselect={handlePlotUnselect}
                  embeddings={embeddings.data}
                  onSelect={handlePlotSelect}
                  color={embeddingsColor}
                  resetIndex={plotResetIndex}
                  showScale
                />
              </Paper>
            </Box>
          </Grid>
          <Grid item xs={4} sx={{ height: '100%' }}>
            {selectedUsers.length > 0 && (
              <Paper sx={{ p: 3, height: '100%', overflow: 'auto' }}>
                <UsersDescription users={selectedUsers} />
              </Paper>
            )}
          </Grid>
        </Grid>
      </Grid>
    </Grid>
  );
}

MetricsEmbeddings.propTypes = {
  // eslint-disable-next-line react/forbid-prop-types
  metricsData: pt.any.isRequired,
};

export default MetricsEmbeddings;
