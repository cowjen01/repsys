import React, { useEffect } from 'react';
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
  toggleFavouriteUser,
} from '../../reducers/root';
import { openRecEditDialog } from '../../reducers/dialogs';
import { RecEditView, RecGridView, RecEditDialog } from '../recommenders';
import { UserPanel, UserSelectDialog } from '../users';
import { ItemDetailDialog } from '../items';

import ConfirmDialog from '../ConfirmDialog';
import { fetchConfig } from '../../reducers/config';
import { fetchUsers } from '../../reducers/users';
import { itemFieldsSelector } from '../../reducers/settings';

function App() {
  const recommenders = useSelector(recommendersSelector);
  const buildMode = useSelector(buildModeSelector);
  const dispatch = useDispatch();
  const itemFields = useSelector(itemFieldsSelector);
  const favouriteUsers = useSelector(favouriteUsersSelector);
  const selectedUser = useSelector(selectedUserSelector);

  useEffect(() => {
    dispatch(fetchConfig());
    dispatch(fetchUsers());
  }, []);

  const handleRecDeleteConfirm = ({ index }) => {
    dispatch(deleteRecommender(index));
  };

  const handleRecommenderAdd = () => {
    dispatch(openRecEditDialog(null));
  };

  const handleFavouriteToggle = () => {
    dispatch(toggleFavouriteUser());
  };

  return (
    <Container maxWidth="xl">
      <Grid container spacing={4}>
        {!buildMode && !itemFields.title && recommenders.length > 0 && (
          <Grid item xs={12}>
            <Alert severity="warning">
              <AlertTitle>Views not configured</AlertTitle>
              It is not configured how the data should be mapped to the view fields. Please open the
              settings in the top-right menu and finish setup.
            </Alert>
          </Grid>
        )}
        {recommenders.length === 0 && (
          <Grid item xs={12}>
            <Alert severity="info">
              <AlertTitle>Recommenders not configured</AlertTitle>
              {!buildMode
                ? 'There are no recommenders, switch to the build mode to create one.'
                : 'There are no recommenders, press the add button to create one.'}
            </Alert>
          </Grid>
        )}
        {recommenders.length !== 0 && (itemFields.title || buildMode) && (
          <>
            <Grid item xs={12} lg={9}>
              <Grid container spacing={3}>
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
            right: 32,
          }}
          onClick={handleRecommenderAdd}
          color="secondary"
        >
          <AddIcon />
        </Fab>
      )}
      {selectedUser && !buildMode && (
        <Fab
          onClick={handleFavouriteToggle}
          color='secondary'
          sx={{
            position: 'absolute',
            bottom: 32,
            right: 32,
          }}
        >
          {!favouriteUsers.includes(selectedUser) ? <FavoriteBorderIcon /> : <FavoriteIcon />}
        </Fab>
      )}
    </Container>
  );
}

export default App;
