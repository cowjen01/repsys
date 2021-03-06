import React, { useState, useEffect, useMemo } from 'react';
import pt from 'prop-types';
import KeyboardArrowRightIcon from '@mui/icons-material/KeyboardArrowRight';
import KeyboardArrowLeftIcon from '@mui/icons-material/KeyboardArrowLeft';
import MemoryIcon from '@mui/icons-material/Memory';
import LabelIcon from '@mui/icons-material/Label';
import { useSelector, useDispatch } from 'react-redux';
import {
  Skeleton,
  Typography,
  Grid,
  Alert,
  AlertTitle,
  Stack,
  IconButton,
  Chip,
} from '@mui/material';

import { ItemCardView } from '../items';
import { openItemDetailDialog } from '../../reducers/dialogs';
import {
  customInteractionsSelector,
  selectedUserSelector,
  interactiveModeSelector,
  addCustomInteraction,
} from '../../reducers/app';
import { recommenderByIndexSelector } from '../../reducers/recommenders';
import { itemViewSelector } from '../../reducers/settings';
import { usePredictItemsByModelMutation } from '../../api';
import ErrorAlert from '../ErrorAlert';
import { capitalize } from '../../utils';

function RecGridView({ index }) {
  const dispatch = useDispatch();
  const customInteractions = useSelector(customInteractionsSelector);
  const selectedUser = useSelector(selectedUserSelector);
  const interactiveMode = useSelector(interactiveModeSelector);
  const itemView = useSelector(itemViewSelector);
  const recommender = useSelector(recommenderByIndexSelector(index));

  const [currentPage, setCurrentPage] = useState(0);

  const { name, model, itemsLimit, modelParams, itemsPerPage } = recommender;

  const [getRecomendations, recommendations] = usePredictItemsByModelMutation();

  useEffect(() => {
    setCurrentPage(0);

    const query = {
      model,
      params: modelParams,
      limit: itemsLimit,
    };

    if (selectedUser) {
      query.user = selectedUser;
    } else {
      query.items = customInteractions.map(({ id }) => id);
    }

    getRecomendations(query);
  }, [selectedUser, customInteractions]);

  const currentBatch = useMemo(() => {
    if (recommendations.data) {
      return recommendations.data.slice(
        itemsPerPage * currentPage,
        itemsPerPage * (currentPage + 1)
      );
    }
    return [];
  }, [currentPage, recommendations.data]);

  const handleItemClick = (item) => {
    if (interactiveMode) {
      dispatch(addCustomInteraction(item));
    } else {
      dispatch(openItemDetailDialog(item));
    }
  };

  const isLastPage =
    recommendations.data &&
    currentPage === Math.ceil(recommendations.data.length / itemsPerPage) - 1;

  const itemWidth = 12 / itemsPerPage;
  const imageHeight = Math.ceil(500 / itemsPerPage);

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
            <Grid container alignItems="center">
              <Grid item sx={{ mr: 2 }}>
                <Typography variant="h6" component="div">
                  {name}
                </Typography>
              </Grid>
              <Grid item>
                <Stack spacing={1} direction="row">
                  <Chip icon={<MemoryIcon />} size="small" label={`Model: ${model}`} />
                  {Object.entries(modelParams)
                    .filter((x) => x[1] !== '')
                    .map(([paramKey, paramValue]) => (
                      <Chip
                        key={paramKey}
                        icon={<LabelIcon />}
                        size="small"
                        label={`${capitalize(paramKey)}: ${paramValue}`}
                      />
                    ))}
                </Stack>
              </Grid>
            </Grid>
          </Grid>
          {recommendations.isSuccess && (
            <Grid item>
              <Stack direction="row">
                <IconButton
                  disabled={currentPage === 0}
                  onClick={() => setCurrentPage(currentPage - 1)}
                >
                  <KeyboardArrowLeftIcon />
                </IconButton>
                <IconButton disabled={isLastPage} onClick={() => setCurrentPage(currentPage + 1)}>
                  <KeyboardArrowRightIcon />
                </IconButton>
              </Stack>
            </Grid>
          )}
        </Grid>
      </Grid>
      <Grid item xs={12}>
        <Grid container spacing={2}>
          {recommendations.isSuccess &&
            currentBatch.map((item) => (
              <Grid key={item.id} item xs={itemWidth}>
                <ItemCardView
                  item={item}
                  imageHeight={imageHeight}
                  onClick={() => handleItemClick(item)}
                />
              </Grid>
            ))}
          {(recommendations.isLoading || recommendations.isUninitialized) &&
            [...Array(itemsPerPage).keys()].map((i) => (
              <Grid key={i} item display="flex" xs={itemWidth}>
                <Skeleton
                  variant="rectangular"
                  height={itemView.image ? imageHeight + 100 : 100}
                  width="100%"
                />
              </Grid>
            ))}
          {recommendations.isSuccess && !recommendations.data.length && (
            <Grid item xs={12}>
              <Alert severity="warning">
                <AlertTitle>No recommended items</AlertTitle>
                The model did not return any recommendations.
              </Alert>
            </Grid>
          )}
          {recommendations.isError && (
            <Grid item xs={12}>
              <ErrorAlert error={recommendations.error} />
            </Grid>
          )}
        </Grid>
      </Grid>
    </Grid>
  );
}

RecGridView.propTypes = {
  index: pt.number.isRequired,
};

export default RecGridView;
