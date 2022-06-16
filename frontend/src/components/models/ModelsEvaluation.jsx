import React from 'react';
import { Grid, LinearProgress } from '@mui/material';
// import { useSelector, useDispatch } from 'react-redux';

import MetricsDistribution from './MetricsDistribution';
import MetricsSummary from './MetricsSummary';
import ErrorAlert from '../ErrorAlert';
import MetricsEmbeddings from './MetricsEmbeddings';
import { useGetModelsMetricsQuery, useGetDatasetQuery } from '../../api';
// import { seenTutorialsSelector } from '../../reducers/app';
// import { openTutorialDialog } from '../../reducers/dialogs';
import TooltipHeader from '../TooltipHeader';

function ModelsEvaluation() {
  const metrics = useGetModelsMetricsQuery();
  const dataset = useGetDatasetQuery();
  // const seenTutorials = useSelector(seenTutorialsSelector);
  // const dispatch = useDispatch();

  // useEffect(() => {
  //   if (!seenTutorials.includes('models') && !metrics.isLoading && !dataset.isLoading) {
  //     dispatch(openTutorialDialog('models'));
  //   }
  // }, [metrics.isLoading, dataset.isLoading]);

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
        <TooltipHeader
          title="Metrics Summary"
          tooltip="The summary shows an average of metrics over the set of test users. Below each value is displayed how the metric changed compared to the previous evaluation results. For details about the metrics, please see the project documentation on GitHub."
        />
        <MetricsSummary metricsData={metrics.data} />
      </Grid>
      <Grid item xs={12}>
        <TooltipHeader
          title="Metrics Distribution"
          tooltip="The histogram shows the distribution of metrics over the set of test users. On the x-axis is the metric value and on the y-axis is the number of users with corresponding metric results. It is possible to select a part of the distribution to display additional information: the position of users in the latent space and the distribution of item attributes they interacted with most. The selection can be canceled using a double-click."
        />
        <MetricsDistribution
          metricsType="user"
          itemAttributes={dataset.data.attributes}
          evaluatedModels={evaluatedModels}
        />
      </Grid>
      <Grid item xs={12}>
        <TooltipHeader
          title="User Embeddings"
          tooltip="The plot shows a visualization of the users, where each point is a test user, and the color corresponds with the measured metric value. It is possible to select a cluster of users using the lasso tool to display 100 of the most popular items within the group and the distribution of attribute values of these items. The selection can be canceled using a double-click."
        />
        <MetricsEmbeddings
          metricsType="user"
          itemAttributes={dataset.data.attributes}
          evaluatedModels={evaluatedModels}
        />
      </Grid>
      {/* <Grid item xs={12}>
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
      </Grid> */}
    </Grid>
  );
}

export default ModelsEvaluation;
