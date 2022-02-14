import React, { useState, useEffect } from 'react';
import pt from 'prop-types';
import KeyboardArrowRightIcon from '@mui/icons-material/KeyboardArrowRight';
import KeyboardArrowLeftIcon from '@mui/icons-material/KeyboardArrowLeft';
import { useSelector, useDispatch } from 'react-redux';
import { Skeleton, Typography, Grid, Alert, AlertTitle, Stack, IconButton } from '@mui/material';

import { ItemCardView } from '../items';
import { openItemDetailDialog } from '../../reducers/dialogs';
import {
  customInteractionsSelector,
  selectedUserSelector,
  sessionRecordingSelector,
  addCustomInteraction,
} from '../../reducers/app';
import { itemViewSelector } from '../../reducers/settings';
import { usePredictItemsByModelMutation } from '../../api';

function RecGridView({ recommender }) {
  const dispatch = useDispatch();
  const customInteractions = useSelector(customInteractionsSelector);
  const selectedUser = useSelector(selectedUserSelector);
  const sessionRecording = useSelector(sessionRecordingSelector);
  const itemView = useSelector(itemViewSelector);

  const [page, setPage] = useState(0);

  const { name, model, itemsLimit, modelParams, itemsPerPage } = recommender;

  const [getRecoms, { isLoading, isSuccess, data, error, isError }] =
    usePredictItemsByModelMutation();

  useEffect(() => {
    const query = {
      model,
      params: modelParams[model],
      limit: itemsLimit,
    };

    if (selectedUser) {
      query.user = selectedUser;
    } else {
      query.interactions = customInteractions.map(({ id }) => id);
    }

    getRecoms(query);
  }, [selectedUser, customInteractions]);

  const handleItemClick = (item) => {
    if (sessionRecording) {
      dispatch(addCustomInteraction(item));
    } else {
      dispatch(
        openItemDetailDialog({
          title: item[itemView.title],
          content: item[itemView.content],
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
              {name}
            </Typography>
          </Grid>
          {isSuccess && (
            <Grid item>
              <Stack direction="row">
                <IconButton disabled={page === 0} onClick={() => setPage(page - 1)}>
                  <KeyboardArrowLeftIcon />
                </IconButton>
                <IconButton
                  disabled={page === Math.ceil(data.length / itemsPerPage) - 1}
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
          {isSuccess &&
            data.slice(itemsPerPage * page, itemsPerPage * (page + 1)).map((item) => (
              <Grid key={item.id} item xs={12} md={12 / itemsPerPage}>
                <ItemCardView
                  item={item}
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
          {isSuccess && data.length === 0 && (
            <Grid item xs={12}>
              <Alert severity="warning">
                <AlertTitle>No recommended items</AlertTitle>
                The model did not return any recommendations.
              </Alert>
            </Grid>
          )}
          {isError && (
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
    name: pt.string,
    itemsPerPage: pt.number,
    itemsLimit: pt.number,
    model: pt.string,
    // eslint-disable-next-line react/forbid-prop-types
    modelParams: pt.any,
  }).isRequired,
};

export default RecGridView;
