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
        <Typography component="div" variant="h6">
          Models Performance
        </Typography>
        <Typography variant="subtitle1" marginBottom={1}>
          A performance in the individual metrics with comparasion to the previous evaluation.
        </Typography>
        <MetricsSummary metricsData={metrics.data} />
      </Grid>
      <Grid item xs={12}>
        <Typography component="div" variant="h6">
          Metrics Distribution
        </Typography>
        <Typography variant="subtitle1" marginBottom={1}>
          A distribution of the metrics for each validation user
        </Typography>
        <UsersDistribution metricsData={metrics.data} />
      </Grid>
      <Grid item xs={12}>
        <Typography component="div" variant="h6">
          Users Embeddings
        </Typography>
        <Typography variant="subtitle1" marginBottom={1}>
          A distribution of the metrics for each validation user
        </Typography>
        <UsersEmbeddings metricsData={metrics.data} />
      </Grid>
    </Grid>
  );
}

export default ModelsEvaluation;
