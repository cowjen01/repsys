import React from 'react';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import DialogContentText from '@mui/material/DialogContentText';
import DialogTitle from '@mui/material/DialogTitle';

function ConfirmDialog({ open, onClose }) {
  return (
    <Dialog
      open={open}
      onClose={() => onClose(false)}
      aria-labelledby="alert-dialog-title"
      aria-describedby="alert-dialog-description"
    >
      <DialogTitle id="alert-dialog-title">Are you sure you want to delete this bar?</DialogTitle>
      <DialogContent>
        <DialogContentText id="alert-dialog-description">
          Deleting this bar all settings will be lost.
        </DialogContentText>
      </DialogContent>
      <DialogActions>
        <Button onClick={() => onClose(false)}>Disagree</Button>
        <Button onClick={() => onClose(true)} autoFocus>
          Agree
        </Button>
      </DialogActions>
    </Dialog>
  );
}

export default ConfirmDialog;
