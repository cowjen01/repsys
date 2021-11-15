import React, { useMemo } from 'react';
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
import { getRequest } from '../api';
import { SettingsDialog } from './settings';
import Snackbar from './Snackbar';
import ConfirmDialog from './ConfirmDialog';

function App() {
  const recommenders = useSelector(recommendersSelector);
  const buildMode = useSelector(buildModeSelector);
  const dispatch = useDispatch();

  const { items: modelsData, isLoading: modelsLoading } = getRequest('/models');

  const defaultModelParams = useMemo(
    () =>
      Object.fromEntries(
        modelsData.map((m) => [m.key, Object.fromEntries(m.params.map((a) => [a.key, a.default]))])
      ),
    [modelsData]
  );

  const handleRecDeleteConfirm = ({ index }) => {
    dispatch(deleteRecommender(index));
  };

  const handleRecommenderEdit = (index) => {
    const params = {
      ...defaultModelParams,
      ...recommenders[index].modelParams,
    };
    dispatch(
      openRecEditDialog({
        ...recommenders[index],
        modelParams: params,
      })
    );
  };

  const handleRecommenderAdd = () => {
    dispatch(
      openRecEditDialog({
        title: 'New bar',
        itemsPerPage: 4,
        itemsLimit: 20,
        model: modelsData[0].key,
        modelParams: defaultModelParams,
      })
    );
  };

  return (
    <Layout>
      <Container maxWidth={!buildMode ? 'xl' : 'lg'}>
        {!buildMode ? (
          <Grid container spacing={4}>
            {recommenders.length === 0 ? (
              <Grid item xs={12}>
                <Typography sx={{ marginTop: 2 }} align="center" variant="h5">
                  There are no recommenders, switch to the Build Mode to create one.
                </Typography>
              </Grid>
            ) : (
              <>
                <Grid item xs={12} lg={9}>
                  <Grid container spacing={3}>
                    {recommenders.map((recommender) => (
                      <Grid item xs={12} key={recommender.id}>
                        <RecGridView recommender={recommender} />
                      </Grid>
                    ))}
                  </Grid>
                </Grid>
                <Grid item xs={12} lg={3}>
                  <UserPanel />
                </Grid>
              </>
            )}
          </Grid>
        ) : (
          <Grid container spacing={3}>
            {recommenders.length === 0 && (
              <Grid item xs={12}>
                <Typography sx={{ marginTop: 2 }} align="center" variant="h5">
                  There are no recommenders, press the add button to create one.
                </Typography>
              </Grid>
            )}
            {recommenders.map((rec, index) => (
              <Grid item xs={12} key={rec.id}>
                <RecEditView
                  modelsLoading={modelsLoading}
                  onEdit={handleRecommenderEdit}
                  title={rec.title}
                  index={index}
                />
              </Grid>
            ))}
          </Grid>
        )}
      </Container>
      <ConfirmDialog onConfirm={handleRecDeleteConfirm} />
      <RecEditDialog models={modelsData} />
      <ItemDetailDialog />
      <UserSelectDialog />
      <Snackbar />
      <SettingsDialog />
      {buildMode && (
        <Fab
          disabled={modelsLoading}
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
