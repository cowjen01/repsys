import React from 'react';
import pt from 'prop-types';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import DialogContentText from '@mui/material/DialogContentText';
import DialogTitle from '@mui/material/DialogTitle';

function ItemDetailDialog({ open, onClose, data }) {
  if (!data) {
    return null;
  }

  return (
    <Dialog open={open} maxWidth="sm" fullWidth onClose={() => onClose(false)}>
      <DialogTitle>{data.title}</DialogTitle>
      <DialogContent>
        <DialogContentText>{data.description || 'No description provided.'}</DialogContentText>
      </DialogContent>
      <DialogActions>
        <Button color="secondary" onClick={onClose} autoFocus>
          Close
        </Button>
      </DialogActions>
    </Dialog>
  );
}

ItemDetailDialog.propTypes = {
  open: pt.bool.isRequired,
  onClose: pt.func.isRequired,
};

export default ItemDetailDialog;
