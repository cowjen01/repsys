import React from 'react';
import pt from 'prop-types';
import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
} from '@mui/material';
import { useDispatch, useSelector } from 'react-redux';

import { closeConfirmDialog, confirmDialogSelector } from '../reducers/dialogs';

function ConfirmDialog({ onConfirm }) {
  const dialog = useSelector(confirmDialogSelector);
  const dispatch = useDispatch();

  const handleClose = () => {
    dispatch(closeConfirmDialog());
  };

  const handleConfirm = () => {
    handleClose();
    onConfirm(dialog.params);
  };

  return (
    <Dialog open={dialog.open} onClose={handleClose} maxWidth="sm" fullWidth>
      <DialogTitle>{dialog.title}</DialogTitle>
      <DialogContent>
        <DialogContentText>{dialog.content}</DialogContentText>
      </DialogContent>
      <DialogActions>
        <Button onClick={handleClose}>Cancel</Button>
        <Button onClick={handleConfirm} autoFocus>
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
