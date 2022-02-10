import React, { useState, useEffect } from 'react';
import pt from 'prop-types';
import KeyboardArrowRightIcon from '@mui/icons-material/KeyboardArrowRight';
import KeyboardArrowLeftIcon from '@mui/icons-material/KeyboardArrowLeft';
import { useSelector, useDispatch } from 'react-redux';
import { Skeleton, Typography, Grid, Alert, AlertTitle, Stack, IconButton } from '@mui/material';

import { ItemCardView } from '../items';
import { fetchPredictions } from './api';
import { openItemDetailDialog } from '../../reducers/dialogs';
import {
  customInteractionsSelector,
  selectedUserSelector,
  sessionRecordingSelector,
  addCustomInteraction,
} from '../../reducers/app';
import { itemFieldsSelector } from '../../reducers/settings';

function RecGridView({ recommender }) {
  const dispatch = useDispatch();
  const customInteractions = useSelector(customInteractionsSelector);
  const selectedUser = useSelector(selectedUserSelector);
  const sessionRecording = useSelector(sessionRecordingSelector);
  const itemFields = useSelector(itemFieldsSelector);

  const [page, setPage] = useState(0);

  const { title, model, itemsLimit, modelParams, itemsPerPage } = recommender;

  const { items, isLoading, error } = fetchPredictions({
    model,
    ...(selectedUser
      ? {
          user: selectedUser,
        }
      : {
          interactions: customInteractions.map((x) => x.id),
        }),
    params: modelParams[model],
    limit: itemsLimit,
  });

  const handleItemClick = (item) => {
    if (sessionRecording) {
      dispatch(addCustomInteraction(item));
    } else {
      dispatch(
        openItemDetailDialog({
          title: item[itemFields.title],
          content: item[itemFields.content],
        })
      );
    }
  };

  useEffect(() => {
    setPage(0);
  }, [selectedUser, customInteractions.length]);

  return (
    <Grid container spacing={1}>
      <Grid item xs={12}>
        <Grid
          container
          alignItems="center"
          justifyContent="space-between"
          sx={{ minHeight: '40px' }}
        >
          <Grid item>
            <Typography variant="h6" component="div">
              {title}
            </Typography>
          </Grid>
          {!isLoading && !error && (
            <Grid item>
              <Stack direction="row">
                <IconButton disabled={page === 0} onClick={() => setPage(page - 1)}>
                  <KeyboardArrowLeftIcon />
                </IconButton>
                <IconButton
                  disabled={page === Math.ceil(items.length / itemsPerPage) - 1}
                  onClick={() => setPage(page + 1)}
                >
                  <KeyboardArrowRightIcon />
                </IconButton>
              </Stack>
            </Grid>
          )}
        </Grid>
      </Grid>
      <Grid item xs={12}>
        <Grid container spacing={2}>
          {!isLoading &&
            !error &&
            items.slice(itemsPerPage * page, itemsPerPage * (page + 1)).map((item) => (
              <Grid key={item.id} item xs={12} md={12 / itemsPerPage}>
                <ItemCardView
                  title={item[itemFields.title]}
                  subtitle={item[itemFields.subtitle]}
                  caption={item[itemFields.caption]}
                  image={item[itemFields.image]}
                  imageHeight={Math.ceil(600 / itemsPerPage)}
                  onClick={() => handleItemClick(item)}
                />
              </Grid>
            ))}
          {isLoading &&
            [...Array(itemsPerPage).keys()].map((i) => (
              <Grid key={i} item display="flex" md={12 / itemsPerPage}>
                <Skeleton
                  variant="rectangular"
                  height={Math.ceil(600 / itemsPerPage) + 50}
                  width="100%"
                />
              </Grid>
            ))}
          {!isLoading && !error && items.length === 0 && (
            <Grid item xs={12}>
              <Alert severity="warning">
                <AlertTitle>No recommended items</AlertTitle>
                The model did not return any recommendations.
              </Alert>
            </Grid>
          )}
          {!isLoading && error && (
            <Grid item xs={12}>
              <Alert severity="error">
                <AlertTitle>API Error</AlertTitle>
                {error}
              </Alert>
            </Grid>
          )}
        </Grid>
      </Grid>
    </Grid>
  );
}

RecGridView.propTypes = {
  recommender: pt.shape({
    title: pt.string,
    itemsPerPage: pt.number,
    itemsLimit: pt.number,
    model: pt.string,
    // eslint-disable-next-line react/forbid-prop-types
    modelParams: pt.any,
  }).isRequired,
};

export default RecGridView;
