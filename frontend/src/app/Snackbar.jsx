import React from 'react';
import MuiSnackbar from '@mui/material/Snackbar';
import Alert from '@mui/material/Alert';
import { useSelector, useDispatch } from 'react-redux';

import { snackbarOpenSelector, snackbarMessageSelector, closeSnackbar } from '../reducers/studio';

function Snackbar() {
  const dispatch = useDispatch();
  const snackbarOpen = useSelector(snackbarOpenSelector);
  const snackbarMessage = useSelector(snackbarMessageSelector);

  const handleClose = () => {
    dispatch(closeSnackbar());
  };

  return (
    <MuiSnackbar
      open={snackbarOpen}
      autoHideDuration={3000}
      onClose={handleClose}
      anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
    >
      <Alert
        elevation={6}
        variant="filled"
        onClose={handleClose}
        severity="success"
        sx={{ width: '100%' }}
      >
        {snackbarMessage}
      </Alert>
    </MuiSnackbar>
  );
}

export default Snackbar;
