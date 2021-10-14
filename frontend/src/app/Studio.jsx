import React, { useState, useMemo } from 'react';
import Container from '@mui/material/Container';
import { useSelector, useDispatch } from 'react-redux';
import Grid from '@mui/material/Grid';
import Fab from '@mui/material/Fab';
import AddIcon from '@mui/icons-material/Add';
import { Typography } from '@mui/material';

import {
  addBar,
  layoutSelector,
  removeBar,
  updateBarsOrder,
  duplicateBar,
  updateBar,
} from './layoutSlice';
import { buildModeSelector, openSnackbar } from './studioSlice';
import ItemBarView from './ItemBarView';
import ItemBarEdit from './ItemBarEdit';
import ConfirmDialog from './ConfirmDialog';
import ItemBarDialog from './ItemBarDialog';
import Layout from './Layout';
import Snackbar from './Snackbar';
import { fetchItems } from './api';

function Studio() {
  const layout = useSelector(layoutSelector);
  const buildMode = useSelector(buildModeSelector);
  const dispatch = useDispatch();

  const [deleteIndex, setDeleteIndex] = useState();
  const [barData, setBarData] = useState();

  const { items: modelData, isLoading: isModelLoading } = fetchItems('/models');
  const { items: userData, isLoading: isUserLoading } = fetchItems('/users');

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
    dispatch(openSnackbar('All updates done!'));
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
        // model: modelsData[0].key,
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
      <Container maxWidth="lg">
        <Grid container spacing={3}>
          {/* <Grid item xs={12}>

          </Grid> */}
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
                  modelAttributes={bar.modelAttributes}
                  itemsPerPage={bar.itemsPerPage}
                />
              )}
            </Grid>
          ))}
          {layout.length === 0 && (
            <Grid item xs={12}>
              <Typography sx={{ marginTop: 2 }} align="center" variant="h5">
                There are currently no bars, press the add button to create one.
              </Typography>
            </Grid>
          )}
        </Grid>
      </Container>
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
