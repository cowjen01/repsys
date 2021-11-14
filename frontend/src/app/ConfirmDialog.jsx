import React from 'react';
import pt from 'prop-types';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import DialogContentText from '@mui/material/DialogContentText';
import DialogTitle from '@mui/material/DialogTitle';
import { useDispatch, useSelector } from 'react-redux';

import { closeConfirmDialog, confirmDialogSelector } from '../reducers/studio';

function ConfirmDialog({ onConfirm }) {
  const dialog = useSelector(confirmDialogSelector);
  const dispatch = useDispatch();

  const handleClose = () => {
    dispatch(closeConfirmDialog());
  };

  const handleConfirm = () => {
    handleClose();
    onConfirm();
  };

  return (
    <Dialog open={dialog.open} onClose={handleClose} maxWidth="sm" fullWidth>
      <DialogTitle>{dialog.title}</DialogTitle>
      <DialogContent>
        <DialogContentText>{dialog.content}</DialogContentText>
      </DialogContent>
      <DialogActions>
        <Button color="secondary" onClick={handleClose}>
          Cancel
        </Button>
        <Button color="secondary" onClick={handleConfirm} autoFocus>
          Confirm
        </Button>
      </DialogActions>
    </Dialog>
  );
}

ConfirmDialog.propTypes = {
  onConfirm: pt.func.isRequired,
};

export default ConfirmDialog;
