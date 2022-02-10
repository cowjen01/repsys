import React from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { Grid, Fab, Container, Alert, AlertTitle } from '@mui/material';
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
} from '../../reducers/root';
import { openRecEditDialog } from '../../reducers/dialogs';
import { UserPanel, UserSelectDialog } from '../users';
import { ItemDetailDialog } from '../items';
import RecEditView from './RecEditView';
import RecGridView from './RecGridView';
import RecEditDialog from './RecEditDialog';
import ConfirmDialog from '../ConfirmDialog';
import { itemFieldsSelector } from '../../reducers/settings';

function RecPreviews() {
  const recommenders = useSelector(recommendersSelector);
  const buildMode = useSelector(buildModeSelector);
  const dispatch = useDispatch();
  const itemFields = useSelector(itemFieldsSelector);
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
    <Container maxWidth="xl">
      <Grid container spacing={4}>
        {!itemFields.title && (
          <Grid item xs={12}>
            <Alert severity="warning">
              <AlertTitle>Views not configured</AlertTitle>
              It is not configured how the data should be mapped to the view fields. Please open the
              settings in the top-right menu and finish setup.
            </Alert>
          </Grid>
        )}
        {(itemFields.title || buildMode) && (
          <>
            <Grid item xs={12} lg={9}>
              <Grid container spacing={3}>
                {recommenders.length === 0 && (
                  <Grid item xs={12}>
                    <Alert severity="info">
                      <AlertTitle>Recommenders not configured</AlertTitle>
                      There are no recommenders, switch to the build mode to create one.
                    </Alert>
                  </Grid>
                )}
                {recommenders.map((recommender, index) =>
                  !buildMode ? (
                    <Grid item xs={12} key={recommender.id}>
                      <RecGridView recommender={recommender} />
                    </Grid>
                  ) : (
                    <Grid item xs={12} key={recommender.id}>
                      <RecEditView title={recommender.title} index={index} />
                    </Grid>
                  )
                )}
              </Grid>
            </Grid>
            <Grid item xs={12} lg={3}>
              <UserPanel />
            </Grid>
          </>
        )}
      </Grid>
      <ConfirmDialog onConfirm={handleRecDeleteConfirm} />
      <RecEditDialog />
      <ItemDetailDialog />
      <UserSelectDialog />
      {buildMode && (
        <Fab
          sx={{
            position: 'absolute',
            bottom: 32,
            left: 32,
          }}
          variant="extended"
          onClick={handleRecommenderAdd}
          color="secondary"
        >
          <AddIcon sx={{ mr: 1 }} />
          Add recommender
        </Fab>
      )}
      {selectedUser && !buildMode && (
        <Fab
          onClick={handleFavouriteToggle}
          variant="extended"
          color="secondary"
          sx={{
            position: 'absolute',
            bottom: 32,
            left: 32,
          }}
        >
          {!favouriteUsers.includes(selectedUser) ? (
            <FavoriteBorderIcon sx={{ mr: 1 }} />
          ) : (
            <FavoriteIcon sx={{ mr: 1 }} />
          )}
          Favourite user
        </Fab>
      )}
    </Container>
  );
}

export default RecPreviews;
