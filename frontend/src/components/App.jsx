import React, { useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { Typography, Grid, Fab, Container } from '@mui/material';
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
import { fetchModels } from '../reducers/models';
import { fetchUsers } from '../reducers/users';

function App() {
  const recommenders = useSelector(recommendersSelector);
  const buildMode = useSelector(buildModeSelector);
  const dispatch = useDispatch();

  useEffect(() => {
    dispatch(fetchModels());
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
      <Container maxWidth={!buildMode ? 'xl' : 'lg'}>
        <Grid container spacing={!buildMode ? 4 : 3}>
          {recommenders.length === 0 ? (
            <Grid item xs={12}>
              <Typography sx={{ marginTop: 2 }} align="center" variant="h5">
                {!buildMode
                  ? 'There are no recommenders, switch to the Build Mode to create one.'
                  : 'There are no recommenders, press the add button to create one.'}
              </Typography>
            </Grid>
          ) : (
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
