import React, { useState } from 'react';
import Container from '@mui/material/Container';
import { useSelector, useDispatch } from 'react-redux';
import Grid from '@mui/material/Grid';
import Fab from '@mui/material/Fab';
import AddIcon from '@mui/icons-material/Add';
import { Typography } from '@mui/material';

import { addBar, layoutSelector, removeBar, updateBarsOrder } from './layoutSlice';
import { buildModeSelector } from './studioSlice';
import ItemBarPreview from './ItemBarPreview';
import ItemBarEdit from './ItemBarEdit';
import ConfirmDialog from './ConfirmDialog';
import Layout from './Layout';

function Studio() {
  const layout = useSelector(layoutSelector);
  const buildMode = useSelector(buildModeSelector);
  const dispatch = useDispatch();

  const [confirmOpen, setConfirmOpen] = useState(false);
  const [deleteIndex, setDeleteIndex] = useState();

  const handleBarMove = (dragIndex, hoverIndex) => {
    dispatch(updateBarsOrder({ dragIndex, hoverIndex }));
  };

  const handleBarDelete = (index) => {
    setConfirmOpen(true);
    setDeleteIndex(index);
  };

  const handleConfirm = (isAgree) => {
    setConfirmOpen(false);

    if (isAgree) {
      dispatch(removeBar(deleteIndex));
    }
  };

  return (
    <Layout>
      <ConfirmDialog open={confirmOpen} onClose={handleConfirm} />
      <Container maxWidth="lg">
        <Grid container spacing={4}>
          {layout.map((bar, index) => (
            <Grid item xs={12} key={bar.id}>
              {buildMode ? (
                <ItemBarEdit
                  onDelete={handleBarDelete}
                  title={bar.title}
                  index={index}
                  onMove={handleBarMove}
                />
              ) : (
                <ItemBarPreview
                  totalItems={bar.totalItems}
                  title={bar.title}
                  itemsPerPage={bar.itemsPerPage}
                />
              )}
            </Grid>
          ))}
          {layout.length === 0 && (
            <Grid item xs={12}>
              <Typography align="center" variant="h5">There are currently no bars, press the add button to create one.</Typography>
            </Grid>
          )}
        </Grid>
      </Container>
      {buildMode && (
        <Fab
          sx={{
            position: 'absolute',
            bottom: 32,
            right: 32,
          }}
          onClick={() => dispatch(addBar('New Bar'))}
          color="primary"
          aria-label="add"
        >
          <AddIcon />
        </Fab>
      )}
    </Layout>
  );
}

export default Studio;
