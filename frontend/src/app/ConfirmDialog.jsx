import React from 'react';
import pt from 'prop-types';
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
      <DialogTitle id="alert-dialog-title">Delete this recommender?</DialogTitle>
      <DialogContent>
        <DialogContentText id="alert-dialog-description">
          Deleting this recommender all settings will be lost.
        </DialogContentText>
      </DialogContent>
      <DialogActions>
        <Button color="secondary" onClick={() => onClose(false)}>
          Cancel
        </Button>
        <Button color="secondary" onClick={() => onClose(true)} autoFocus>
          Delete
        </Button>
      </DialogActions>
    </Dialog>
  );
}

ConfirmDialog.propTypes = {
  open: pt.bool.isRequired,
  onClose: pt.func.isRequired,
};

export default ConfirmDialog;
