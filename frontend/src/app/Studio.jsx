import React, { useState, useMemo } from 'react';
import Container from '@mui/material/Container';
import { useSelector, useDispatch } from 'react-redux';
import Grid from '@mui/material/Grid';
import Fab from '@mui/material/Fab';
import AddIcon from '@mui/icons-material/Add';
import { Typography } from '@mui/material';
import Drawer from '@mui/material/Drawer';

import {
  addBar,
  recommendersSelector,
  removeBar,
  updateBarsOrder,
  duplicateBar,
  updateBar,
} from '../reducers/recommenders';
import {
  buildModeSelector,
  openSnackbar,
  setSelectedUser,
  selectedUserSelector,
  sessionRecordSelector,
} from '../reducers/studio';
import RecommenderView from './RecommenderView';
import RecommenderEdit from './RecommenderEdit';
import ConfirmDialog from './ConfirmDialog';
import RecommenderDialog from './RecommenderDialog';
import Layout from './Layout';
import Snackbar from './Snackbar';
import ModelMetrics from './ModelMetrics';
import UserPanel from './UserPanel';
import UserSearch from './UserSearch';
import ItemDetailDialog from './ItemDetailDialog';
import { getRequest } from './api';

function Studio() {
  const recommenders = useSelector(recommendersSelector);
  const buildMode = useSelector(buildModeSelector);
  const selectedUser = useSelector(selectedUserSelector);
  const sessionRecord = useSelector(sessionRecordSelector);
  const dispatch = useDispatch();

  const [customInteractions, setCustomInteractions] = useState([]);
  const [deleteIndex, setDeleteIndex] = useState();
  const [barData, setBarData] = useState();
  const [itemData, setItemData] = useState();
  const [userSearchOpen, setUserSearchOpen] = useState(false);
  const [metricsOpen, setMetricsOpen] = useState(false);

  const { items: modelData, isLoading: isModelLoading } = getRequest('/models');

  const defaultModelParams = useMemo(
    () =>
      Object.fromEntries(
        modelData.map((m) => [m.key, Object.fromEntries(m.params.map((a) => [a.key, a.default]))])
      ),
    [modelData]
  );

  const handleBarMove = (dragIndex, hoverIndex) => {
    dispatch(updateBarsOrder({ dragIndex, hoverIndex }));
  };

  const handleBarDelete = (index) => {
    setDeleteIndex(index);
  };

  const handleDeleteConfirm = (isAgree) => {
    if (isAgree) {
      dispatch(removeBar(deleteIndex));
    }
    setDeleteIndex(undefined);
  };

  const handleBarDuplicate = (index) => {
    dispatch(duplicateBar(index));
    // if (index === layout.length - 1) {
    //   divRef.current.scrollIntoView({ behavior: 'smooth', block: 'end', inline: 'nearest' });
    // }
  };

  const handleItemClick = (data) => {
    if (sessionRecord) {
      setCustomInteractions((arr) => [data, ...arr]);
    } else {
      setItemData(data);
    }
  };

  const handleItemDetailClose = () => {
    setItemData(undefined);
  };

  const handleBarDialogClose = () => {
    setBarData(undefined);
  };

  const handleBarSubmit = (values) => {
    const data = {
      ...values,
      itemsPerPage: parseInt(values.itemsPerPage, 10),
    };
    if (!values.id) {
      dispatch(addBar(data));
    } else {
      dispatch(updateBar(data));
    }
    dispatch(openSnackbar('All settings applied!'));
    handleBarDialogClose();
  };

  const handleBarEdit = (index) => {
    if (!isModelLoading) {
      // TODO: deep merge of model params
      const params = {
        ...defaultModelParams,
        ...recommenders[index].modelParams,
      };
      setBarData({
        ...recommenders[index],
        modelParams: params,
      });
    }
  };

  const handleBarAdd = () => {
    if (!isModelLoading) {
      setBarData({
        title: 'New bar',
        itemsPerPage: 4,
        itemsLimit: 20,
        model: modelData[0].key,
        modelParams: defaultModelParams,
      });
    }
  };

  return (
    <Layout>
      <ConfirmDialog open={deleteIndex !== undefined} onClose={handleDeleteConfirm} />
      <RecommenderDialog
        open={barData !== undefined}
        initialValues={barData}
        models={modelData}
        onClose={handleBarDialogClose}
        onSubmit={handleBarSubmit}
      />
      <ItemDetailDialog
        open={itemData !== undefined}
        data={itemData}
        onClose={handleItemDetailClose}
      />
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
                    {recommenders.map((rec) => (
                      <Grid item xs={12} key={rec.id}>
                        <RecommenderView
                          title={rec.title}
                          model={rec.model}
                          selectedUser={selectedUser}
                          customInteractions={customInteractions}
                          onMetricsClick={() => setMetricsOpen(true)}
                          modelParams={rec.modelParams}
                          itemsPerPage={rec.itemsPerPage}
                          itemsLimit={rec.itemsLimit}
                          onItemClick={handleItemClick}
                        />
                      </Grid>
                    ))}
                  </Grid>
                </Grid>
                <Grid item xs={12} lg={3}>
                  <UserPanel
                    selectedUser={selectedUser}
                    onInteractionsDelete={() => {
                      setCustomInteractions([]);
                    }}
                    onUserSelect={(user) => {
                      dispatch(setSelectedUser(user));
                      setCustomInteractions([]);
                    }}
                    onSearchClick={() => setUserSearchOpen(true)}
                    customInteractions={customInteractions}
                  />
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
                <RecommenderEdit
                  onDuplicate={handleBarDuplicate}
                  onDelete={handleBarDelete}
                  onEdit={handleBarEdit}
                  title={rec.title}
                  index={index}
                  onMove={handleBarMove}
                />
              </Grid>
            ))}
          </Grid>
        )}
      </Container>
      <Drawer anchor="bottom" open={metricsOpen} onClose={() => setMetricsOpen(false)}>
        <ModelMetrics />
      </Drawer>
      <Drawer anchor="right" open={userSearchOpen} onClose={() => setUserSearchOpen(false)}>
        <UserSearch
          customInteractions={customInteractions}
          onUserSelect={(user) => {
            dispatch(setSelectedUser(user));
            setCustomInteractions([]);
            setUserSearchOpen(false);
          }}
          onInteractionsSelect={(interactions) => {
            setUserSearchOpen(false);
            setCustomInteractions(interactions);
            dispatch(setSelectedUser(null));
          }}
        />
      </Drawer>
      <Snackbar />
      {buildMode && (
        <Fab
          sx={{
            position: 'absolute',
            bottom: 32,
            right: 32,
          }}
          onClick={handleBarAdd}
          color="secondary"
          aria-label="add"
        >
          <AddIcon />
        </Fab>
      )}
    </Layout>
  );
}

export default Studio;
