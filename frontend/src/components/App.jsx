import React, { useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { Grid, Fab, Container, Alert, AlertTitle } from '@mui/material';
import AddIcon from '@mui/icons-material/Add';

import { recommendersSelector, deleteRecommender } from '../reducers/recommenders';
import { buildModeSelector } from '../reducers/root';
import { openRecEditDialog } from '../reducers/dialogs';
import { RecEditView, RecGridView, RecEditDialog } from './recommenders';
import { UserPanel, UserSelectDialog } from './users';
import { ItemDetailDialog } from './items';
import Layout from './Layout';
import { SettingsDialog } from './settings';
import Snackbar from './Snackbar';
import ConfirmDialog from './ConfirmDialog';
import { fetchConfig } from '../reducers/config';
import { fetchUsers } from '../reducers/users';
import { itemFieldsSelector } from '../reducers/settings';

function App() {
  const recommenders = useSelector(recommendersSelector);
  const buildMode = useSelector(buildModeSelector);
  const dispatch = useDispatch();
  const itemFields = useSelector(itemFieldsSelector);

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

  return (
    <Layout>
      <Container maxWidth={buildMode && recommenders.length !== 0 ? 'lg' : 'xl'}>
        <Grid container spacing={!buildMode ? 4 : 3}>
          {!buildMode && !itemFields.title && recommenders.length > 0 && (
            <Grid item xs={12}>
              <Alert severity="warning">
                <AlertTitle>Views not configured</AlertTitle>
                It is not configured how the data should be mapped to the view fields. Please open
                the settings in the top-left menu and finish setup.
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
              <Grid item xs={12} lg={!buildMode ? 9 : 12}>
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
              {!buildMode && (
                <Grid item xs={12} lg={3}>
                  <UserPanel />
                </Grid>
              )}
            </>
          )}
        </Grid>
      </Container>
      <ConfirmDialog onConfirm={handleRecDeleteConfirm} />
      <RecEditDialog />
      <ItemDetailDialog />
      <UserSelectDialog />
      <Snackbar />
      <SettingsDialog />
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
    </Layout>
  );
}

export default App;
