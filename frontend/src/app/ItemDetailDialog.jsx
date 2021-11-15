import React from 'react';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import DialogContentText from '@mui/material/DialogContentText';
import DialogTitle from '@mui/material/DialogTitle';
import { useDispatch, useSelector } from 'react-redux';

import { closeItemDetailDialog, itemDetailDialogSelector } from '../reducers/dialogs';

function ItemDetailDialog() {
  const dialog = useSelector(itemDetailDialogSelector);
  const dispatch = useDispatch();

  const handleClose = () => {
    dispatch(closeItemDetailDialog());
  };

  return (
    <Dialog open={dialog.open} maxWidth="sm" fullWidth onClose={handleClose}>
      <DialogTitle>{dialog.title}</DialogTitle>
      <DialogContent>
        <DialogContentText>{dialog.content}</DialogContentText>
      </DialogContent>
      <DialogActions>
        <Button color="secondary" onClick={handleClose} autoFocus>
          Close
        </Button>
      </DialogActions>
    </Dialog>
  );
}

export default ItemDetailDialog;
