import React from 'react';
import { Grid, Typography, LinearProgress } from '@mui/material';

import MetricsDistribution from './MetricsDistribution';
import MetricsSummary from './MetricsSummary';
import ErrorAlert from '../ErrorAlert';
import MetricsEmbeddings from './MetricsEmbeddings';
import { useGetModelsMetricsQuery, useGetDatasetQuery } from '../../api';

function ModelsEvaluation() {
  const metrics = useGetModelsMetricsQuery();
  const dataset = useGetDatasetQuery();

  if (metrics.isLoading || dataset.isLoading) {
    return <LinearProgress />;
  }

  if (metrics.isError) {
    return <ErrorAlert error={metrics.error} />;
  }

  if (dataset.isError) {
    return <ErrorAlert error={dataset.error} />;
  }

  const evaluatedModels = Object.keys(metrics.data.results);

  return (
    <Grid container spacing={4}>
      <Grid item xs={12}>
        <Typography component="div" variant="h6" gutterBottom>
          Metrics Summary
        </Typography>
        <MetricsSummary metricsData={metrics.data} />
      </Grid>
      <Grid item xs={12}>
        <Typography component="div" variant="h6" gutterBottom>
          User Metrics Distribution
        </Typography>
        <MetricsDistribution
          metricsType="user"
          itemAttributes={dataset.data.attributes}
          evaluatedModels={evaluatedModels}
        />
      </Grid>
      <Grid item xs={12}>
        <Typography component="div" variant="h6" gutterBottom>
          User Metrics Embeddings
        </Typography>
        <MetricsEmbeddings
          metricsType="user"
          itemAttributes={dataset.data.attributes}
          evaluatedModels={evaluatedModels}
        />
      </Grid>
      <Grid item xs={12}>
        <Typography component="div" variant="h6" gutterBottom>
          Item Metrics Distribution
        </Typography>
        <MetricsDistribution
          metricsType="item"
          itemAttributes={dataset.data.attributes}
          evaluatedModels={evaluatedModels}
        />
      </Grid>
      <Grid item xs={12}>
        <Typography component="div" variant="h6" gutterBottom>
          Item Metrics Embeddings
        </Typography>
        <MetricsEmbeddings
          metricsType="item"
          itemAttributes={dataset.data.attributes}
          evaluatedModels={evaluatedModels}
        />
      </Grid>
    </Grid>
  );
}

export default ModelsEvaluation;
