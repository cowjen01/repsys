import React, { useState, useEffect } from 'react';
import pt from 'prop-types';
import KeyboardArrowRightIcon from '@mui/icons-material/KeyboardArrowRight';
import KeyboardArrowLeftIcon from '@mui/icons-material/KeyboardArrowLeft';
import { useSelector, useDispatch } from 'react-redux';
import { Typography, Grid, Skeleton, Alert, AlertTitle, Stack, IconButton } from '@mui/material';

import { ItemCardView } from '../items';
import { postRequest } from '../../api';
import { openItemDetailDialog } from '../../reducers/dialogs';
import {
  customInteractionsSelector,
  selectedUserSelector,
  sessionRecordingSelector,
  addCustomInteraction,
} from '../../reducers/root';

function RecGridView({ recommender }) {
  const dispatch = useDispatch();
  const customInteractions = useSelector(customInteractionsSelector);
  const selectedUser = useSelector(selectedUserSelector);
  const sessionRecording = useSelector(sessionRecordingSelector);

  const [page, setPage] = useState(0);

  const { title, model, itemsLimit, modelParams, itemsPerPage } = recommender;

  const { items, isLoading } = postRequest('/predict', {
    model,
    ...(selectedUser
      ? {
          user: selectedUser.id,
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
          title: item.title,
          content: item.description,
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
        <Grid container alignItems="center" justifyContent="space-between">
          <Grid item>
            {/* <Stack direction="row" spacing={1}>
              <Chip
                // color="primary"
                size="small"
                // clickable
                // variant="outlined"
                icon={<LightbulbIcon />}
                label={model}
              />
            </Stack> */}
            <Typography variant="h6" component="div">
              {title}
            </Typography>
          </Grid>
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
        </Grid>
      </Grid>
      <Grid item xs={12}>
        <Grid container spacing={2}>
          {!isLoading
            ? items.slice(itemsPerPage * page, itemsPerPage * (page + 1)).map((item) => (
                <Grid key={item.id} item xs={12} md={12 / itemsPerPage}>
                  <ItemCardView
                    title={item.title}
                    subtitle={item.subtitle}
                    caption={item.caption}
                    image={item.image}
                    imageHeight={Math.ceil(600 / itemsPerPage)}
                    onClick={() => handleItemClick(item)}
                  />
                </Grid>
              ))
            : [...Array(itemsPerPage).keys()].map((i) => (
                <Grid key={i} item display="flex" md={12 / itemsPerPage}>
                  <Skeleton
                    variant="rectangular"
                    height={Math.ceil(600 / itemsPerPage) + 100}
                    width="100%"
                  />
                </Grid>
              ))}
          {!isLoading && items.length === 0 && (
            <Grid item xs={12}>
              <Alert severity="warning">
                <AlertTitle>No recommended items</AlertTitle>
                The model has not returned any recommendations.
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
