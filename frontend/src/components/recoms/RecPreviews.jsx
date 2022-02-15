import React from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { Grid, Fab, Alert, AlertTitle } from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import FavoriteIcon from '@mui/icons-material/Favorite';
import FavoriteBorderIcon from '@mui/icons-material/FavoriteBorder';

import { recommendersSelector, deleteRecommender } from '../../reducers/recommenders';
import {
  buildModeSelector,
  favouriteUsersSelector,
  selectedUserSelector,
  addUserToFavourites,
  removeUserFromFavourites,
} from '../../reducers/app';
import { openRecEditDialog } from '../../reducers/dialogs';
import ControlPanel from './ControlPanel';
import SelectorDialog from './SelectorDialog';
import ItemDetailDialog from './ItemDetailDialog';
import RecEditView from './RecEditView';
import RecGridView from './RecGridView';
import RecEditDialog from './RecEditDialog';
import ConfirmDialog from '../ConfirmDialog';
import { itemViewSelector } from '../../reducers/settings';

const fabStyles = {
  position: 'absolute',
  bottom: 32,
  left: 32,
};

function RecPreviews() {
  const dispatch = useDispatch();
  const recommenders = useSelector(recommendersSelector);
  const buildMode = useSelector(buildModeSelector);
  const itemView = useSelector(itemViewSelector);
  const favouriteUsers = useSelector(favouriteUsersSelector);
  const selectedUser = useSelector(selectedUserSelector);

  const handleRecDeleteConfirm = ({ index }) => {
    dispatch(deleteRecommender(index));
  };

  const handleRecommenderAdd = () => {
    dispatch(openRecEditDialog(null));
  };

  const handleFavouriteToggle = () => {
    if (favouriteUsers.includes(selectedUser)) {
      dispatch(removeUserFromFavourites(selectedUser));
    } else {
      dispatch(addUserToFavourites(selectedUser));
    }
  };

  return (
    <>
      <Grid container spacing={4}>
        {!itemView.title && (
          <Grid item xs={12}>
            <Alert severity="warning">
              <AlertTitle>Views not configured</AlertTitle>
              It is not configured how the data should be mapped to the views. Please open the
              settings in the top-right menu and finish the setup.
            </Alert>
          </Grid>
        )}
        {(itemView.title || buildMode) && (
          <>
            <Grid item xs={12} lg={9}>
              <Grid container spacing={3}>
                {!recommenders.length && (
                  <Grid item xs={12}>
                    <Alert severity="info">
                      <AlertTitle>Recommenders not configured</AlertTitle>
                      There have been no recommenders created yet. Please switch to the build mode
                      and create one.
                    </Alert>
                  </Grid>
                )}
                {recommenders.map((recommender, index) =>
                  !buildMode ? (
                    <Grid item xs={12} key={recommender.name}>
                      <RecGridView index={index} />
                    </Grid>
                  ) : (
                    <Grid item xs={12} key={recommender.name}>
                      <RecEditView name={recommender.name} index={index} />
                    </Grid>
                  )
                )}
              </Grid>
            </Grid>
            <Grid item xs={12} lg={3}>
              <ControlPanel />
            </Grid>
          </>
        )}
      </Grid>
      <ConfirmDialog onConfirm={handleRecDeleteConfirm} />
      <RecEditDialog />
      <ItemDetailDialog />
      <SelectorDialog />
      {buildMode && (
        <Fab sx={fabStyles} variant="extended" onClick={handleRecommenderAdd} color="secondary">
          <AddIcon sx={{ mr: 1 }} />
          Add recommender
        </Fab>
      )}
      {selectedUser && !buildMode && (
        <Fab onClick={handleFavouriteToggle} variant="extended" color="secondary" sx={fabStyles}>
          {!favouriteUsers.includes(selectedUser) ? (
            <FavoriteBorderIcon sx={{ mr: 1 }} />
          ) : (
            <FavoriteIcon sx={{ mr: 1 }} />
          )}
          Favourite user
        </Fab>
      )}
    </>
  );
}

export default RecPreviews;
