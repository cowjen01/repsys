import React, { useState, useMemo } from 'react';
import Container from '@mui/material/Container';
import { useSelector, useDispatch } from 'react-redux';
import Grid from '@mui/material/Grid';
import Fab from '@mui/material/Fab';
import AddIcon from '@mui/icons-material/Add';
import { Typography } from '@mui/material';
import Drawer from '@mui/material/Drawer';
import Alert from '@mui/material/Alert';
import AlertTitle from '@mui/material/AlertTitle';

import {
  addBar,
  layoutSelector,
  removeBar,
  updateBarsOrder,
  duplicateBar,
  updateBar,
} from './layoutSlice';
import {
  buildModeSelector,
  openSnackbar,
  setSelectedUser,
  selectedUserSelector,
} from './studioSlice';
import ItemBarView from './ItemBarView';
import ItemBarEdit from './ItemBarEdit';
import ConfirmDialog from './ConfirmDialog';
import ItemBarDialog from './ItemBarDialog';
import Layout from './Layout';
import Snackbar from './Snackbar';
import ModelMetrics from './ModelMetrics';
import UserPanel from './UserPanel';
import UserSearch from './UserSearch';
import { fetchItems } from './api';

function Studio() {
  const layout = useSelector(layoutSelector);
  const buildMode = useSelector(buildModeSelector);
  const selectedUser = useSelector(selectedUserSelector);
  const dispatch = useDispatch();

  const [testInteractions, setTestInteractions] = useState(null);
  const [deleteIndex, setDeleteIndex] = useState();
  const [barData, setBarData] = useState();
  const [userSearchOpen, setUserSearchOpen] = useState(false);
  const [metricsOpen, setMetricsOpen] = useState(false);

  const { items: modelData, isLoading: isModelLoading } = fetchItems('/models');

  const modelAttributes = useMemo(
    () =>
      Object.fromEntries(
        modelData.map((m) => [
          m.key,
          Object.fromEntries(m.attributes.map((a) => [a.key, a.defaultValue || ''])),
        ])
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
      setBarData(layout[index]);
    }
  };

  const handleBarAdd = () => {
    if (!isModelLoading) {
      setBarData({
        title: 'New bar',
        itemsPerPage: 4,
        model: modelData[0].key,
        modelAttributes,
      });
    }
  };

  return (
    <Layout>
      <ConfirmDialog open={deleteIndex !== undefined} onClose={handleDeleteConfirm} />
      <ItemBarDialog
        open={barData !== undefined}
        initialValues={barData}
        models={modelData}
        onClose={handleBarDialogClose}
        onSubmit={handleBarSubmit}
      />
      <Container maxWidth={!buildMode ? 'xl' : 'lg'}>
        {!buildMode ? (
          <Grid container spacing={4}>
            {layout.length === 0 ? (
              <Grid item xs={12}>
                <Typography sx={{ marginTop: 2 }} align="center" variant="h5">
                  There are no recommenders, switch to the Build Mode to create one.
                </Typography>
              </Grid>
            ) : (
              <>
                <Grid item xs={12} lg={9}>
                  {selectedUser ? (
                    <Grid container spacing={3}>
                      {layout.map((bar, index) => (
                        <Grid item xs={12} key={bar.id}>
                          {buildMode ? (
                            <ItemBarEdit
                              onDuplicate={handleBarDuplicate}
                              onDelete={handleBarDelete}
                              onEdit={handleBarEdit}
                              title={bar.title}
                              index={index}
                              onMove={handleBarMove}
                            />
                          ) : (
                            <ItemBarView
                              title={bar.title}
                              model={bar.model}
                              user={selectedUser}
                              onMetricsClick={() => setMetricsOpen(true)}
                              modelAttributes={bar.modelAttributes}
                              itemsPerPage={bar.itemsPerPage}
                            />
                          )}
                        </Grid>
                      ))}
                    </Grid>
                  ) : (
                    <>
                      <Typography variant="h6" component="div" gutterBottom>
                        User Recommendations
                      </Typography>
                      <Alert severity="warning" sx={{ marginTop: 0 }}>
                        <AlertTitle>Empty User</AlertTitle>
                        Select a user from the right panel to see some recommendations.
                      </Alert>
                    </>
                  )}
                </Grid>
                <Grid item xs={12} lg={3}>
                  <UserPanel
                    selectedUser={selectedUser}
                    onInteractionsDelete={() => {
                      setTestInteractions(null);
                    }}
                    onUserSelect={(user) => {
                      dispatch(setSelectedUser(user));
                      setTestInteractions(null);
                    }}
                    onSearchClick={() => setUserSearchOpen(true)}
                    testInteractions={testInteractions}
                  />
                </Grid>
              </>
            )}
          </Grid>
        ) : (
          <Grid container spacing={3}>
            {layout.length === 0 && (
              <Grid item xs={12}>
                <Typography sx={{ marginTop: 2 }} align="center" variant="h5">
                  There are no recommenders, press the Add button to create one.
                </Typography>
              </Grid>
            )}
            {layout.map((bar, index) => (
              <Grid item xs={12} key={bar.id}>
                <ItemBarEdit
                  onDuplicate={handleBarDuplicate}
                  onDelete={handleBarDelete}
                  onEdit={handleBarEdit}
                  title={bar.title}
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
          onUserSelect={(user) => {
            dispatch(setSelectedUser(user));
            setTestInteractions(null);
            setUserSearchOpen(false);
          }}
          onInteractionsSelect={(interactions) => {
            setUserSearchOpen(false);
            setTestInteractions(interactions);
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
