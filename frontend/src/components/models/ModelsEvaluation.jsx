import React from 'react';
import { Grid, Typography, LinearProgress } from '@mui/material';

import UsersDistribution from './UsersDistribution';
import MetricsSummary from './MetricsSummary';
import ErrorAlert from '../ErrorAlert';
import UsersEmbeddings from './UsersEmbeddings';
import { useGetModelsMetricsQuery } from '../../api';

function ModelsEvaluation() {
  const metrics = useGetModelsMetricsQuery();

  if (metrics.isLoading) {
    return <LinearProgress />;
  }

  if (metrics.isError) {
    return <ErrorAlert error={metrics.error} />;
  }
  return (
    <Grid container spacing={4}>
      <Grid item xs={12}>
        <Typography component="div" variant="h6" gutterBottom>
          Models Performance
        </Typography>
        <MetricsSummary metricsData={metrics.data} />
      </Grid>
      <Grid item xs={12}>
        <Typography component="div" variant="h6" gutterBottom>
          Metrics Distribution
        </Typography>
        <UsersDistribution metricsData={metrics.data} />
      </Grid>
      <Grid item xs={12}>
        <Typography component="div" variant="h6" gutterBottom>
          Users Embeddings
        </Typography>
        <UsersEmbeddings metricsData={metrics.data} />
      </Grid>
    </Grid>
  );
}

export default ModelsEvaluation;
