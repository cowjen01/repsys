import React from 'react';
import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  Typography,
} from '@mui/material';
import { useDispatch, useSelector } from 'react-redux';

import { closeItemDetailDialog, itemDetailDialogSelector } from '../../reducers/dialogs';
import { itemViewSelector } from '../../reducers/settings';

function ItemDetailDialog() {
  const dialog = useSelector(itemDetailDialogSelector);
  const dispatch = useDispatch();
  const itemView = useSelector(itemViewSelector);

  const handleClose = () => {
    dispatch(closeItemDetailDialog());
  };

  return (
    <Dialog open={dialog.open} maxWidth="sm" fullWidth onClose={handleClose}>
      {dialog.item && (
        <>
          <DialogTitle>{dialog.item[itemView.title]}</DialogTitle>
          <DialogContent>
            <DialogContentText>
              {/* <Typography gutterBottom>
                {dialog.item[itemView.subtitle]}
              </Typography>
              <Typography gutterBottom>
                {dialog.item[itemView.caption]}
              </Typography> */}
              <Typography gutterBottom>
                {dialog.item[itemView.content] || 'No description provided.'}
              </Typography>
            </DialogContentText>
          </DialogContent>
        </>
      )}
      <DialogActions>
        <Button color="secondary" onClick={handleClose} autoFocus>
          Close
        </Button>
      </DialogActions>
    </Dialog>
  );
}

export default ItemDetailDialog;
