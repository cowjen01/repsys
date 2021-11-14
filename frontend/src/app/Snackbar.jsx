import React from 'react';
import MuiSnackbar from '@mui/material/Snackbar';
import Alert from '@mui/material/Alert';
import { useSelector, useDispatch } from 'react-redux';

import { snackbarSelector, closeSnackbar } from '../reducers/studio';

function Snackbar() {
  const dispatch = useDispatch();
  const snackbar = useSelector(snackbarSelector);

  const handleClose = () => {
    dispatch(closeSnackbar());
  };

  return (
    <MuiSnackbar
      open={snackbar.open}
      autoHideDuration={3000}
      onClose={handleClose}
      anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
    >
      <Alert
        elevation={6}
        variant="filled"
        onClose={handleClose}
        severity={snackbar.severity}
        sx={{ width: '100%' }}
      >
        {snackbar.message}
      </Alert>
    </MuiSnackbar>
  );
}

export default Snackbar;
