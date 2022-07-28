import React from 'react';
import { LinearProgress } from '@mui/material';
import { useParams } from 'react-router-dom';

import ErrorAlert from '../ErrorAlert';
import { useGetModelsMetricsQuery, useGetDatasetQuery } from '../../api';
import MetricsDistribution from '../models/MetricsDistribution';
import MetricsEmbeddings from '../models/MetricsEmbeddings';

function EvaluationWidget() {
  const metrics = useGetModelsMetricsQuery();
  const dataset = useGetDatasetQuery();
  const { formatType } = useParams();

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

  if (formatType === 'histogram') {
    return (
      <MetricsDistribution
        metricsType="user"
        itemAttributes={dataset.data.attributes}
        evaluatedModels={evaluatedModels}
      />
    );
  }

  if (formatType === 'embeddings') {
    return (
      <MetricsEmbeddings
        metricsType="user"
        displayVisualSettings={false}
        itemAttributes={dataset.data.attributes}
        evaluatedModels={evaluatedModels}
      />
    );
  }

  return null;
}

export default EvaluationWidget;
