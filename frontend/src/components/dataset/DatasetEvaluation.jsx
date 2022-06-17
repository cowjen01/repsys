import React from 'react';
import { Grid, LinearProgress } from '@mui/material';
// import { useSelector, useDispatch } from 'react-redux';

import ErrorAlert from '../ErrorAlert';
import { useGetDatasetQuery } from '../../api';
import ItemsEmbeddings from './ItemsEmbeddings';
import UsersEmbeddings from './UsersEmbeddings';
// import { seenTutorialsSelector } from '../../reducers/app';
// import { openTutorialDialog } from '../../reducers/dialogs';
import TooltipHeader from '../TooltipHeader';

function DatasetEvaluation() {
  const dataset = useGetDatasetQuery();
  // const seenTutorials = useSelector(seenTutorialsSelector);
  // const dispatch = useDispatch();

  // useEffect(() => {
  //   if (!seenTutorials.includes('dataset') && !dataset.isLoading) {
  //     dispatch(openTutorialDialog('dataset'));
  //   }
  // }, [dataset.isLoading]);

  if (dataset.isLoading) {
    return <LinearProgress />;
  }

  if (dataset.isError) {
    return <ErrorAlert error={dataset.error} />;
  }

  return (
    <Grid container spacing={4}>
      <Grid item xs={12}>
        <TooltipHeader
          title="Item Embeddings"
          tooltip="The plot shows a visualization of all items from the catalog. It is possible to filter the items based on their attributes or select a cluster of them to see the distribution of their attributes. The selection can be canceled using a double-click."
        />
        <ItemsEmbeddings attributes={dataset.data.attributes} />
      </Grid>
      <Grid item xs={12}>
        <TooltipHeader
          title="User Embeddings"
          tooltip="The plot shows a visualization of the training users. It is possible to filter them based on the attributes of items the users interacted with by specifying the minimum interactions made with such items or selecting a cluster of users to display 100 of the most popular items within the group and the distribution of attribute values of these items. To apply the minimum interactions filter, please click outside the field after changing the value. The selection can be canceled using a double-click."
        />
        <UsersEmbeddings attributes={dataset.data.attributes} />
      </Grid>
    </Grid>
  );
}

export default DatasetEvaluation;
